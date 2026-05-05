"""
Traffic Analysis Demo - Flask Backend
Run: bash launch.sh
Access via OSC OnDemand: https://ondemand.osc.edu/node/hostname/5000/
No ffmpeg needed — frames served as JPEGs and animated in browser canvas
"""

import os
import cv2
import json
import uuid
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from torch_geometric.nn import GATConv

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR   = os.path.join(BASE_DIR, 'demo')
MODELS_DIR = os.path.join(BASE_DIR, 'detection', 'models')
UPLOAD_DIR = os.path.join(DEMO_DIR, 'uploads')
OUTPUT_DIR = os.path.join(DEMO_DIR, 'outputs')
YOLO_CKPT  = os.path.join(MODELS_DIR, 'yolo11x_visdrone_50epoch.pt')
GNN_CKPT   = os.path.join(MODELS_DIR, 'gnn_routing.pt')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
WEIGHTS = {
    'car': 1.0, 'van': 1.5, 'truck': 3.0,
    'bus': 3.0, 'motor': 0.5, 'bicycle': 0.3
}
N_CLUSTERS       = 20
KNN_K            = 3
WINDOW_SIZE      = 10
HIDDEN_DIM       = 128
NUM_LAYERS       = 4
GAT_HEADS        = 4
FRAME_STEP       = 5    # save every Nth frame for canvas animation (~6fps for 30fps video)
USE_MACROBLOCKS  = True  # Toggle between old k-means and new macroblock clustering
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── GAT Model ──────────────────────────────────────────────────────────────────
class TrafficGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, heads=4):
        super().__init__()
        self.convs      = nn.ModuleList()
        self.bns        = nn.ModuleList()
        self.residuals  = nn.ModuleList()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=0.2))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.residuals.append(nn.Identity())
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=0.2))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.residuals.append(nn.Identity())
        self.out_conv = GATConv(hidden_dim, hidden_dim // 2, heads=1, concat=False, dropout=0.1)
        self.out_fc   = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.input_proj(x))
        for conv, bn, res in zip(self.convs, self.bns, self.residuals):
            x_in = x
            x    = conv(x, edge_index)
            x    = bn(x)
            x    = F.elu(x)
            x    = F.dropout(x, p=0.2, training=self.training)
            x    = x + res(x_in)
        x = self.out_conv(x, edge_index)
        x = F.elu(x)
        x = self.out_fc(x)
        return torch.sigmoid(x).squeeze(-1)

# ── Load models at startup ─────────────────────────────────────────────────────
print("Loading YOLO11x...")
try:
    yolo_model = YOLO(YOLO_CKPT)
    print(f"YOLO loaded — {len(yolo_model.names)} classes")
except FileNotFoundError:
    print("YOLO model not found, using dummy detection")
    yolo_model = None

gat_model = None
if os.path.exists(GNN_CKPT):
    print("Loading TrafficGAT...")
    try:
        gat_model = TrafficGAT(WINDOW_SIZE, HIDDEN_DIM, NUM_LAYERS, GAT_HEADS).to(DEVICE)
        gat_model.load_state_dict(torch.load(GNN_CKPT, map_location=DEVICE))
        gat_model.eval()
        print("GAT loaded.")
    except Exception as e:
        print(f"GAT model loading failed: {e}")
        gat_model = None
else:
    print(f"WARNING: GAT checkpoint not found at {GNN_CKPT}")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=DEMO_DIR)

class PrefixMiddleware:
    def __init__(self, wsgi_app):
        self.app = wsgi_app

    def __call__(self, environ, start_response):
        import re
        path  = environ.get('PATH_INFO', '')
        match = re.match(r'^(/node/[^/]+/\d+)(.*)', path)
        if match:
            environ['PATH_INFO']   = match.group(2) or '/'
            environ['SCRIPT_NAME'] = match.group(1)
        return self.app(environ, start_response)

app.wsgi_app = PrefixMiddleware(app.wsgi_app)

def perform_clustering(all_vehicle_data, all_detections, use_macroblocks=True):
    """
    Perform clustering using either traditional k-means or new macroblock approach.
    Returns: (cluster_centers, zone_info, comparison_stats, macroblock_data)

    all_detections: list of per-frame detection lists, used for congestion normalisation.
    """
    # Total vehicle count across all frames — used to normalise congestion scores
    total_vehicle_count = sum(len(frame) for frame in all_detections)

    if not use_macroblocks:
        # ── Traditional K-Means Clustering ─────────────────────────────────────
        all_centers = [v['center'] for v in all_vehicle_data]
        n_clusters = min(N_CLUSTERS, len(all_centers)) if all_centers else 0

        if n_clusters < 2:
            return None, None, {'error': 'Not enough detections'}, None

        centers_array = np.array(all_centers)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(centers_array)
        cluster_centers = kmeans.cluster_centers_

        zone_info = [{
            'centroid': center,
            'vehicles': [],
            'congestion': 0,
            'type': 'traditional'
        } for center in cluster_centers]

        comparison_stats = {
            'algorithm': 'traditional',
            'method': 'traditional',
            'n_zones': n_clusters,
            'n_macroblocks': None,
            'subdivided_zones': 0,
            'combined_zones': 0,
            'zone_types': ['traditional'] * n_clusters,
            'zone_counts': {
                'total': n_clusters,
                'subdivided': 0,
                'combined': 0
            },
            'congestion_metrics': {
                'average': 0.0,
                'max': 0.0
            },
            'processing_details': {
                'iterations': 1,
                'thresholds': {
                    'congestion': 0.0
                }
            }
        }

        return cluster_centers, zone_info, comparison_stats, None

    else:
        # ── Macroblock-based Clustering ───────────────────────────────────────
        if len(all_vehicle_data) < 2:
            return None, None, {'error': 'Not enough detections'}, None

        # Step 1: Create initial macroblocks
        n_macroblocks = max(3, min(8, len(all_vehicle_data) // 10))
        centers_array = np.array([v['center'] for v in all_vehicle_data])

        kmeans_macro = KMeans(n_clusters=n_macroblocks, random_state=42, n_init=10)
        macro_labels = kmeans_macro.fit_predict(centers_array)

        # Step 2: Calculate weighted centroids and congestion
        # FIX: normalise by total vehicle count (not frame count) so congestion
        # stays in [0, 1] and the subdivision threshold is meaningful.
        macroblock_data = []
        CONGESTION_THRESHOLD_HIGH = 0.7
        CONGESTION_THRESHOLD_LOW  = 0.2

        for m in range(n_macroblocks):
            vehicles_in_macro = [v for v, label in zip(all_vehicle_data, macro_labels) if label == m]

            if not vehicles_in_macro:
                continue

            total_weight = sum(v['weight'] for v in vehicles_in_macro)
            weighted_centroid = np.average(
                [v['center'] for v in vehicles_in_macro],
                weights=[v['weight'] for v in vehicles_in_macro],
                axis=0
            )
            # Normalise against total vehicle weight across all frames
            total_all_weight = sum(v['weight'] for v in all_vehicle_data)
            congestion_level = total_weight / total_all_weight if total_all_weight > 0 else 0.0

            macroblock_data.append({
                'id': m,
                'centroid': weighted_centroid,
                'vehicles': vehicles_in_macro,
                'congestion': congestion_level,
                'type': 'macroblock'
            })

        # Step 3: Subdivide high-congestion macroblocks
        final_zones    = []
        subdivided_count = 0
        total_all_weight = sum(v['weight'] for v in all_vehicle_data)

        for macro in macroblock_data:
            if macro['congestion'] > CONGESTION_THRESHOLD_HIGH and len(macro['vehicles']) > 5:
                n_subzones = min(3, len(macro['vehicles']) // 3)
                if n_subzones >= 2:
                    sub_centers = np.array([v['center'] for v in macro['vehicles']])
                    kmeans_sub  = KMeans(n_clusters=n_subzones, random_state=42, n_init=10)
                    sub_labels  = kmeans_sub.fit_predict(sub_centers)

                    for s in range(n_subzones):
                        sub_vehicles = [v for v, label in zip(macro['vehicles'], sub_labels) if label == s]
                        if sub_vehicles:
                            sub_centroid = np.average(
                                [v['center'] for v in sub_vehicles],
                                weights=[v['weight'] for v in sub_vehicles],
                                axis=0
                            )
                            final_zones.append({
                                'centroid':     sub_centroid,
                                'vehicles':     sub_vehicles,
                                'congestion':   sum(v['weight'] for v in sub_vehicles) / total_all_weight if total_all_weight > 0 else 0.0,
                                'type':         'subdivided',
                                'parent_macro': macro['id']
                            })
                    subdivided_count += 1
                else:
                    final_zones.append(macro)
            else:
                final_zones.append(macro)

        # Step 4: Combine nearby low-congestion zones
        merged_zones   = []
        processed      = set()
        combined_count = 0

        for i, zone1 in enumerate(final_zones):
            if i in processed:
                continue

            current_vehicles = zone1['vehicles'].copy()
            current_centroid = zone1['centroid']
            current_type     = zone1['type']

            for j, zone2 in enumerate(final_zones):
                if j in processed or j == i:
                    continue

                if zone2['congestion'] < CONGESTION_THRESHOLD_LOW:
                    distance = np.linalg.norm(zone1['centroid'] - zone2['centroid'])
                    if distance < 200:
                        current_vehicles.extend(zone2['vehicles'])
                        if current_vehicles:
                            current_centroid = np.average(
                                [v['center'] for v in current_vehicles],
                                weights=[v['weight'] for v in current_vehicles],
                                axis=0
                            )
                        current_type = 'combined' if current_type != 'subdivided' else current_type
                        processed.add(j)
                        combined_count += 1

            merged_zones.append({
                'centroid':   current_centroid,
                'vehicles':   current_vehicles,
                'congestion': sum(v['weight'] for v in current_vehicles) / total_all_weight if total_all_weight > 0 else 0.0,
                'type':       current_type
            })
            processed.add(i)

        # cluster_centers reflects the final merged zones
        cluster_centers = np.array([zone['centroid'] for zone in merged_zones])

        comparison_stats = {
            'algorithm':        'macroblock',
            'method':           'macroblock',
            'n_zones':          len(merged_zones),
            'n_macroblocks':    n_macroblocks,
            'subdivided_zones': subdivided_count,
            'combined_zones':   combined_count,
            'zone_types':       [zone['type'] for zone in merged_zones],
            'zone_counts': {
                'total':      len(merged_zones),
                'subdivided': subdivided_count,
                'combined':   combined_count
            },
            'congestion_metrics': {
                'average': float(np.mean([zone['congestion'] for zone in merged_zones])),
                'max':     float(np.max( [zone['congestion'] for zone in merged_zones]))
            },
            'processing_details': {
                'iterations': 1,
                'thresholds': {
                    'congestion': CONGESTION_THRESHOLD_HIGH
                }
            }
        }

        # Return merged_zones as zone_info; macroblock_data kept separately for
        # graph construction (congestion-aware edge colouring).
        return cluster_centers, merged_zones, comparison_stats, macroblock_data

@app.route('/')
def index():
    return send_from_directory(DEMO_DIR, 'index.html')

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(DEMO_DIR, filename)

@app.route('/process', methods=['POST'])
def process_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        session_id = str(uuid.uuid4())[:8]
        input_path = os.path.join(UPLOAD_DIR, f'{session_id}_input.mp4')
        video_file.save(input_path)

        orig_dir = os.path.join(OUTPUT_DIR, f'{session_id}_orig')
        yolo_dir = os.path.join(OUTPUT_DIR, f'{session_id}_yolo')
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(yolo_dir, exist_ok=True)

        print(f"[{session_id}] Processing {video_file.filename}")

        # ── Extract frames and run YOLO ────────────────────────────────────────
        cap    = cv2.VideoCapture(input_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_detections  = []
        orig_frame_urls = []
        yolo_frame_urls = []
        bg_frame        = None
        frame_idx       = 0
        saved_idx       = 0

        print(f"[{session_id}] Running YOLO on {total} frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STEP == 0:
                orig_name = f'{saved_idx:04d}.jpg'
                cv2.imwrite(os.path.join(orig_dir, orig_name), frame)
                orig_frame_urls.append(f'outputs/{session_id}_orig/{orig_name}')

                if frame_idx == (total // 2 // FRAME_STEP) * FRAME_STEP:
                    bg_frame = frame.copy()

            if yolo_model is not None:
                results    = yolo_model.predict(source=frame, conf=0.25, save=False, verbose=False)[0]
                ann_frame  = frame.copy()
                frame_dets = []

                for box in results.boxes:
                    cls   = int(box.cls[0].item())
                    label = yolo_model.names[cls]
                    conf  = box.conf[0].item()
                    x1, y1, x2, y2 = [int(v.item()) for v in box.xyxy[0]]
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    if label in WEIGHTS:
                        frame_dets.append({
                            'label': label, 'cx': cx, 'cy': cy,
                            'weight': WEIGHTS[label]
                        })

                    cv2.rectangle(ann_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(ann_frame, f'{label} {conf:.2f}',
                                (x1, max(y1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                ann_frame  = frame.copy()
                frame_dets = []
                for i in range(np.random.randint(5, 15)):
                    cx    = np.random.randint(50, width - 50)
                    cy    = np.random.randint(50, height - 50)
                    label = np.random.choice(list(WEIGHTS.keys()))
                    frame_dets.append({'label': label, 'cx': cx, 'cy': cy, 'weight': WEIGHTS[label]})
                    x1, y1 = cx - 20, cy - 10
                    x2, y2 = cx + 20, cy + 10
                    cv2.rectangle(ann_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(ann_frame, f'{label} dummy',
                                (x1, max(y1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if frame_idx % FRAME_STEP == 0:
                yolo_name = f'{saved_idx:04d}.jpg'
                cv2.imwrite(os.path.join(yolo_dir, yolo_name), ann_frame)
                yolo_frame_urls.append(f'outputs/{session_id}_yolo/{yolo_name}')
                saved_idx += 1

            all_detections.append(frame_dets)
            frame_idx += 1

        cap.release()
        print(f"[{session_id}] Processed {frame_idx} frames, saved {saved_idx} for animation.")

        bg_url = None
        if bg_frame is not None:
            bg_path = os.path.join(OUTPUT_DIR, f'{session_id}_bg.jpg')
            cv2.imwrite(bg_path, bg_frame)
            bg_url = f'outputs/{session_id}_bg.jpg'

        # ── Clustering (Traditional or Macroblock) ─────────────────────────────
        all_vehicle_data = []
        for frame_dets in all_detections:
            for det in frame_dets:
                all_vehicle_data.append({
                    'center': [det['cx'], det['cy']],
                    'weight': det['weight']
                })

        result = perform_clustering(all_vehicle_data, all_detections, USE_MACROBLOCKS)
        if result[0] is None:
            return jsonify({'error': result[2]['error']}), 400

        cluster_centers, zone_info, comparison_stats, macroblock_data = result
        n_clusters = len(cluster_centers)  # always reflects final merged zone count

        print(f"[{session_id}] Clustering complete:")
        print(f"  Method: {comparison_stats['method']}")
        print(f"  Zones: {comparison_stats['n_zones']}")
        if comparison_stats['n_macroblocks']:
            print(f"  Macroblocks: {comparison_stats['n_macroblocks']}")
            print(f"  Subdivided: {comparison_stats['subdivided_zones']}")
            print(f"  Combined: {comparison_stats['combined_zones']}")
            print(f"  Zone types: {comparison_stats['zone_types']}")

        # ── Build KNN graph ────────────────────────────────────────────────────
        # Both paths now operate on cluster_centers / zone_info (merged zones),
        # keeping n_clusters consistent throughout.
        G = nx.Graph()
        for i in range(n_clusters):
            G.add_node(i, pos=[float(cluster_centers[i][0]), float(cluster_centers[i][1])])

        if USE_MACROBLOCKS and macroblock_data:
            # Group adjacent highly-congested raw macroblocks for edge colouring
            congested_groups = []
            mb_processed = set()

            for i, macro1 in enumerate(macroblock_data):
                if i in mb_processed or macro1['congestion'] < 0.7:
                    continue
                group = [macro1]
                mb_processed.add(i)
                for j, macro2 in enumerate(macroblock_data):
                    if j in mb_processed or macro2['congestion'] < 0.7:
                        continue
                    if np.linalg.norm(macro1['centroid'] - macro2['centroid']) < 300:
                        group.append(macro2)
                        mb_processed.add(j)
                if len(group) > 1:
                    congested_groups.append(group)

            # KNN edges on merged zones with congestion-aware weights
            for i in range(n_clusters):
                dists = []
                for j in range(n_clusters):
                    if i == j:
                        continue
                    distance     = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    base_weight  = 1.0 / (distance + 1e-6)
                    cong_i       = zone_info[i]['congestion']
                    cong_j       = zone_info[j]['congestion']
                    penalty      = 0.7 if (cong_i > 0.7 or cong_j > 0.7) else 1.0
                    dists.append((base_weight * 1.2 * penalty, j))
                dists.sort(reverse=True)
                for weight, j in dists[:KNN_K]:
                    if not G.has_edge(i, j):
                        cong_i = zone_info[i]['congestion']
                        cong_j = zone_info[j]['congestion']
                        is_congested = cong_i > 0.7 or cong_j > 0.7
                        G.add_edge(i, j, weight=weight,
                                   color='red' if is_congested else 'white',
                                   congested=is_congested)
        else:
            # Traditional: simple inverse-distance KNN
            for i in range(n_clusters):
                dists = []
                for j in range(n_clusters):
                    if i == j:
                        continue
                    distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    dists.append((1.0 / (distance + 1e-6), j))
                dists.sort(reverse=True)
                for weight, j in dists[:KNN_K]:
                    if not G.has_edge(i, j):
                        G.add_edge(i, j, weight=weight, color='white', congested=False)

        # MST fallback for connectivity
        if not nx.is_connected(G):
            G_full = nx.Graph()
            G_full.add_nodes_from(range(n_clusters))
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    G_full.add_edge(i, j, weight=distance)
            mst = nx.minimum_spanning_tree(G_full, algorithm='kruskal')
            for u, v in mst.edges():
                if not G.has_edge(u, v):
                    distance = np.linalg.norm(cluster_centers[u] - cluster_centers[v])
                    G.add_edge(u, v, weight=1.0 / (distance + 1e-6), color='white', congested=False)

        # ── Build congestion time series ───────────────────────────────────────
        raw_scores = np.zeros((n_clusters, len(all_detections)))

        if USE_MACROBLOCKS:
            for zone_idx in range(n_clusters):
                for t, frame_vehicles in enumerate(all_detections):
                    zone_congestion = 0
                    for det in frame_vehicles:
                        vehicle_center     = np.array([det['cx'], det['cy']])
                        distances_to_zones = [np.linalg.norm(vehicle_center - cluster_centers[z])
                                              for z in range(n_clusters)]
                        if np.argmin(distances_to_zones) == zone_idx:
                            zone_congestion += det['weight']
                    raw_scores[zone_idx, t] = zone_congestion
        else:
            for t, frame_dets in enumerate(all_detections):
                for det in frame_dets:
                    dists   = [np.linalg.norm(np.array([det['cx'], det['cy']]) - cluster_centers[z])
                               for z in range(n_clusters)]
                    nearest = int(np.argmin(dists))
                    raw_scores[nearest, t] += det['weight']

        max_val = raw_scores.max()
        if max_val > 0:
            raw_scores = raw_scores / max_val

        # ── Generate animated graph overlay frames ─────────────────────────────
        graph_dir        = os.path.join(OUTPUT_DIR, f'{session_id}_graph')
        graph_frame_urls = []
        os.makedirs(graph_dir, exist_ok=True)

        all_vals = raw_scores.flatten()
        p33_val  = float(np.percentile(all_vals, 33))
        p66_val  = float(np.percentile(all_vals, 66))

        def zone_bgr(score):
            if score > p66_val: return (0,  50, 220)
            if score > p33_val: return (0, 165, 255)
            return                     (80, 200,  0)

        graph_edges_list = list(G.edges())

        for saved_i in range(len(orig_frame_urls)):
            t         = min(saved_i * FRAME_STEP, raw_scores.shape[1] - 1)
            orig_path = os.path.join(orig_dir, f'{saved_i:04d}.jpg')
            frame     = cv2.imread(orig_path)
            if frame is None:
                continue

            for u, v in graph_edges_list:
                p1         = (int(cluster_centers[u][0]), int(cluster_centers[u][1]))
                p2         = (int(cluster_centers[v][0]), int(cluster_centers[v][1]))
                edge_color = G.get_edge_data(u, v, {}).get('color', 'white')
                color      = (0, 0, 255) if edge_color == 'red' else (255, 255, 255)
                cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

            zone_scores = raw_scores[:, t]
            for z in range(n_clusters):
                cx, cy = int(cluster_centers[z][0]), int(cluster_centers[z][1])
                color  = zone_bgr(zone_scores[z])
                cv2.circle(frame, (cx, cy), 16, color, -1)
                cv2.circle(frame, (cx, cy), 16, (255, 255, 255), 2, cv2.LINE_AA)
                label = f'Z{z}'
                cv2.putText(frame, label, (cx - 9, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            graph_name = f'{saved_i:04d}.jpg'
            cv2.imwrite(os.path.join(graph_dir, graph_name), frame)
            graph_frame_urls.append(f'outputs/{session_id}_graph/{graph_name}')

        print(f"[{session_id}] Graph overlay frames generated.")

        # ── GAT inference ──────────────────────────────────────────────────────
        edges      = list(G.edges())
        edge_index = torch.tensor(
            [[u, v] for u, v in edges] + [[v, u] for u, v in edges],
            dtype=torch.long
        ).t().contiguous().to(DEVICE)

        if gat_model is not None and raw_scores.shape[1] >= WINDOW_SIZE:
            congestion_t = torch.tensor(raw_scores, dtype=torch.float).to(DEVICE)
            window       = congestion_t[:, -WINDOW_SIZE:]
            with torch.no_grad():
                gnn_scores = gat_model(window, edge_index).cpu().numpy()
        else:
            gnn_scores = raw_scores.mean(axis=1)

        # ── Scale coordinates to 800x600 canvas ───────────────────────────────
        pad     = 60
        xs      = cluster_centers[:, 0]
        ys      = cluster_centers[:, 1]
        x_range = xs.max() - xs.min() if xs.max() != xs.min() else 1
        y_range = ys.max() - ys.min() if ys.max() != ys.min() else 1

        nodes = []
        for i in range(n_clusters):
            nodes.append({
                'id':         i,
                'x':          float(pad + (cluster_centers[i][0] - xs.min()) / x_range * (800 - 2*pad)),
                'y':          float(pad + (cluster_centers[i][1] - ys.min()) / y_range * (600 - 2*pad)),
                'congestion': float(gnn_scores[i]),
                'size':       15
            })

        graph_edges = [{'src': int(u), 'dst': int(v)} for u, v in G.edges()]
        adj   = defaultdict(list)
        adj_b = defaultdict(list)
        for u, v in G.edges():
            adj[str(u)].append({'node': v, 'weight': float(gnn_scores[v])})
            adj[str(v)].append({'node': u, 'weight': float(gnn_scores[u])})
            adj_b[str(u)].append({'node': v, 'weight': 1.0})
            adj_b[str(v)].append({'node': u, 'weight': 1.0})

        print(f"[{session_id}] Done. {n_clusters} zones, {len(graph_edges)} edges.")

        return jsonify({
            'session_id':       session_id,
            'orig_frames':      orig_frame_urls,
            'yolo_frames':      yolo_frame_urls,
            'graph_frames':     graph_frame_urls,
            'bg_image':         bg_url,
            'fps':              round(fps / FRAME_STEP, 1),
            'clustering_stats': comparison_stats,
            'graph': {
                'nodes':        nodes,
                'edges':        graph_edges,
                'adj_gnn':      dict(adj),
                'adj_baseline': dict(adj_b),
                'n_nodes':      n_clusters
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"Starting Traffic Analysis Demo")
    print(f"Device  : {DEVICE}")
    print(f"YOLO    : {YOLO_CKPT}")
    print(f"GAT     : {GNN_CKPT}")
    print(f"Access  : http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
// Traffic Analysis Demo
// 4 steps: YOLO detection → live zone animation → static graph → interactive routing

class TrafficDemo {
    constructor() {
        this.selectedNodes = { start: null, destination: null };
        this.graphData      = null;
        this.clusteringStats = null;
        this.videoFile      = null;
        this.sessionId      = null;
        this.bgImage        = null;
        this.activeVersion  = 'original'; // 'original' | 'macroblock'

        // Step 1 animation (original + yolo)
        this.origFrames  = [];
        this.yoloFrames  = [];
        this.animFrame   = 0;
        this.animTimer   = null;
        this.animFps     = 6;

        // Step 2 animation (graph overlay)
        this.graphFrames    = [];
        this.graphAnimFrame = 0;
        this.graphAnimTimer = null;

        this.initializeEventListeners();
    }

    setVersion(version) {
        if (this.activeVersion === version) return;
        this.activeVersion = version;

        // Update toggle buttons
        document.getElementById('btnOriginal').classList.toggle('active',   version === 'original');
        document.getElementById('btnMacroblock').classList.toggle('active', version === 'macroblock');

        // Reset any in-progress results so the user knows to re-process
        const sections = ['video-comparison', 'graph-animation', 'routing-demo'];
        sections.forEach(id => document.getElementById(id).classList.add('hidden'));
        this.graphData       = null;
        this.clusteringStats = null;
        this.origFrames      = [];
        this.yoloFrames      = [];
        this.graphFrames     = [];
        this.stopAnimation();
        this.stopGraphAnimation();

        this.showStatus(
            `Switched to ${version === 'macroblock' ? 'Macroblock' : 'Original'} mode. ` +
            `Re-process your video to apply.`,
            'info'
        );
    }

    initializeEventListeners() {
        document.getElementById('processBtn').addEventListener('click',         () => this.processVideo());
        document.getElementById('videoInput').addEventListener('change',        (e) => this.handleFileSelect(e));
        document.getElementById('nextStepBtn').addEventListener('click',        () => this.showGraphAnimation());
        document.getElementById('nextStepBtn2').addEventListener('click',       () => this.showRoutingDemo());
        document.getElementById('calculateRoutesBtn').addEventListener('click', () => this.calculateRoutes());
        document.getElementById('clearSelectionBtn').addEventListener('click',  () => this.clearSelection());

        // Step 1 sync controls
        document.getElementById('syncPlayBtn').addEventListener('click',  () => this.startAnimation());
        document.getElementById('syncPauseBtn').addEventListener('click', () => this.stopAnimation());
        document.getElementById('syncResetBtn').addEventListener('click', () => this.resetAnimation());

        // Step 2 graph animation controls
        document.getElementById('graphPlayBtn').addEventListener('click',  () => this.startGraphAnimation());
        document.getElementById('graphPauseBtn').addEventListener('click', () => this.stopGraphAnimation());
        document.getElementById('graphResetBtn').addEventListener('click', () => this.resetGraphAnimation());

        // Canvas clicks
        document.getElementById('routingGraph').addEventListener('click',  (e) => this.handleCanvasClick(e));
    }

    handleFileSelect(event) {
        const files = event.target.files;
        if (files.length > 0) {
            this.videoFile = files[0];
            this.showStatus(`File selected: ${this.videoFile.name}`, 'success');
        }
    }

    getBase() {
        const path = window.location.pathname;
        return path.endsWith('/') ? path : path + '/';
    }

    // ── Process video ───────────────────────────────────────────────────────────
    async processVideo() {
        if (!this.videoFile) {
            this.showStatus('Please select a video file first.', 'error');
            return;
        }

        this.showStatus('Processing video with YOLO... This may take a moment.', 'info');
        this.showProgressBar(true);
        document.getElementById('processBtn').disabled = true;

        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 8;
            if (progress > 88) progress = 88;
            this.updateProgressBar(progress);
        }, 400);

        try {
            const formData = new FormData();
            formData.append('video', this.videoFile);
            formData.append('version', this.activeVersion);
            // Tell the backend which model weights to load
            formData.append('model_weights',
                this.activeVersion === 'macroblock'
                    ? 'gnn_routing_best.pt'
                    : 'gnn_routing_best_previous.pt'
            );

            const response = await fetch('process', { method: 'POST', body: formData });

            if (!response.ok) {
                let errMsg = response.statusText;
                try { const e = await response.json(); errMsg = e.error || errMsg; } catch(_) {}
                throw new Error(errMsg);
            }

            const data     = await response.json();
            this.sessionId      = data.session_id;
            this.graphData      = data.graph;
            this.clusteringStats = data.clustering_stats || null;
            this.animFps        = data.fps || 6;

            // Load background image for static graph canvas
            if (data.bg_image) {
                this.bgImage     = new Image();
                this.bgImage.src = this.getBase() + data.bg_image;
            }

            // Preload all frame sets
            this.showStatus('Loading frames...', 'info');
            await this.preloadAllFrames(data.orig_frames, data.yolo_frames, data.graph_frames);

            this.showVideoComparison();
            this.startAnimation();
            this.showStatus('Video processed successfully!', 'success');

        } catch (error) {
            this.showStatus(`Processing failed: ${error.message}`, 'error');
        } finally {
            clearInterval(progressInterval);
            this.updateProgressBar(100);
            setTimeout(() => this.showProgressBar(false), 600);
            document.getElementById('processBtn').disabled = false;
        }
    }

    async preloadAllFrames(origUrls, yoloUrls, graphUrls) {
        const base = this.getBase();
        const load = url => new Promise(resolve => {
            const img   = new Image();
            img.onload  = () => resolve(img);
            img.onerror = () => resolve(null);
            img.src     = base + url;
        });

        const [origImages, yoloImages, graphImages] = await Promise.all([
            Promise.all(origUrls.map(load)),
            Promise.all(yoloUrls.map(load)),
            Promise.all((graphUrls || []).map(load))
        ]);

        this.origFrames  = origImages.filter(Boolean);
        this.yoloFrames  = yoloImages.filter(Boolean);
        this.graphFrames = graphImages.filter(Boolean);
        this.animFrame      = 0;
        this.graphAnimFrame = 0;
    }

    // ── Step 1 animation (original + YOLO side by side) ─────────────────────────
    startAnimation() {
        this.stopAnimation();
        const interval = Math.round(1000 / this.animFps);
        this.animTimer = setInterval(() => {
            if (this.origFrames.length === 0) return;
            const origCanvas = document.getElementById('originalCanvas');
            const yoloCanvas = document.getElementById('yoloCanvas');
            if (!origCanvas || !yoloCanvas) return;

            // Resize canvas to match image aspect ratio on first frame
            const img = this.origFrames[0];
            if (img && origCanvas.dataset.sized !== '1') {
                const aspect = img.naturalHeight / img.naturalWidth;
                origCanvas.height = Math.round(origCanvas.width * aspect);
                yoloCanvas.width  = origCanvas.width;
                yoloCanvas.height = origCanvas.height;
                origCanvas.dataset.sized = '1';
            }

            const origImg = this.origFrames[this.animFrame % this.origFrames.length];
            const yoloImg = this.yoloFrames[this.animFrame % this.yoloFrames.length];
            const oCtx    = origCanvas.getContext('2d');
            const yCtx    = yoloCanvas.getContext('2d');

            if (origImg) oCtx.drawImage(origImg, 0, 0, origCanvas.width, origCanvas.height);
            if (yoloImg) yCtx.drawImage(yoloImg, 0, 0, yoloCanvas.width, yoloCanvas.height);

            this.animFrame = (this.animFrame + 1) % this.origFrames.length;
        }, interval);
    }

    stopAnimation() {
        if (this.animTimer) { clearInterval(this.animTimer); this.animTimer = null; }
    }

    resetAnimation() {
        this.stopAnimation();
        this.animFrame = 0;
        const origCanvas = document.getElementById('originalCanvas');
        const yoloCanvas = document.getElementById('yoloCanvas');
        if (origCanvas) origCanvas.getContext('2d').clearRect(0, 0, origCanvas.width, origCanvas.height);
        if (yoloCanvas) yoloCanvas.getContext('2d').clearRect(0, 0, yoloCanvas.width, yoloCanvas.height);
    }

    // ── Step 2 animation (graph overlay) ────────────────────────────────────────
    startGraphAnimation() {
        this.stopGraphAnimation();
        const interval    = Math.round(1000 / this.animFps);
        this.graphAnimTimer = setInterval(() => {
            if (this.graphFrames.length === 0) return;
            const canvas = document.getElementById('graphAnimCanvas');
            if (!canvas) return;

            const img = this.graphFrames[0];
            if (img && canvas.dataset.sized !== '1') {
                const aspect  = img.naturalHeight / img.naturalWidth;
                canvas.height = Math.round(canvas.width * aspect);
                canvas.dataset.sized = '1';
            }

            const frame = this.graphFrames[this.graphAnimFrame % this.graphFrames.length];
            if (frame) {
                const ctx = canvas.getContext('2d');
                ctx.drawImage(frame, 0, 0, canvas.width, canvas.height);
            }

            this.graphAnimFrame = (this.graphAnimFrame + 1) % this.graphFrames.length;
        }, interval);
    }

    stopGraphAnimation() {
        if (this.graphAnimTimer) { clearInterval(this.graphAnimTimer); this.graphAnimTimer = null; }
    }

    resetGraphAnimation() {
        this.stopGraphAnimation();
        this.graphAnimFrame = 0;
        const canvas = document.getElementById('graphAnimCanvas');
        if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    }

    // ── Section navigation — sections stay visible, scroll into view ────────────
    showVideoComparison() {
        document.getElementById('video-comparison').classList.remove('hidden');
        document.getElementById('video-comparison').scrollIntoView({ behavior: 'smooth' });
    }

    displayClusteringStats() {
        // Stats box removed from UI
    }

    showGraphAnimation() {
        document.getElementById('graph-animation').classList.remove('hidden');
        document.getElementById('graph-animation').scrollIntoView({ behavior: 'smooth' });
        this.displayClusteringStats();
        setTimeout(() => this.startGraphAnimation(), 200);
    }

    showGraphAnalysis() {
        document.getElementById('graph-analysis').classList.remove('hidden');
        document.getElementById('graph-analysis').scrollIntoView({ behavior: 'smooth' });
        setTimeout(() => this.drawGraph(), 150);
    }

    showRoutingDemo() {
        document.getElementById('routing-demo').classList.remove('hidden');
        document.getElementById('routing-demo').scrollIntoView({ behavior: 'smooth' });
        setTimeout(() => this.drawRoutingGraph(), 150);
    }

    // ── Graph drawing helpers ───────────────────────────────────────────────────
    getPercentileThresholds() {
        const scores = this.graphData.nodes.map(n => n.congestion);
        const sorted = [...scores].sort((a, b) => a - b);
        return {
            p33: sorted[Math.floor(sorted.length * 0.33)],
            p66: sorted[Math.floor(sorted.length * 0.66)],
            min: sorted[0],
            max: sorted[sorted.length - 1]
        };
    }

    nodeColor(node, p33, p66) {
        if (this.selectedNodes.start === node.id)       return '#3498db';
        if (this.selectedNodes.destination === node.id) return '#9b59b6';
        return node.congestion > p66 ? '#e74c3c' :
               node.congestion > p33 ? '#f39c12' : '#2ecc71';
    }

    drawBackground(ctx, canvas) {
        if (this.bgImage && this.bgImage.complete && this.bgImage.naturalWidth > 0) {
            ctx.globalAlpha = 0.5;
            ctx.drawImage(this.bgImage, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
        } else {
            ctx.fillStyle = '#1a1a2e';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }
    }

    drawEdges(ctx, gnnPath = null, baselinePath = null) {
        this.graphData.edges.forEach(({ src, dst }) => {
            const a = this.graphData.nodes[src];
            const b = this.graphData.nodes[dst];
            if (!a || !b) return;
            const onGnn  = gnnPath      && this.edgeOnPath(gnnPath,      src, dst);
            const onBase = baselinePath && this.edgeOnPath(baselinePath,  src, dst);
            ctx.setLineDash([]);
            if (onGnn)       { ctx.strokeStyle = '#2ecc71'; ctx.lineWidth = 5; }
            else if (onBase) { ctx.strokeStyle = '#e74c3c'; ctx.lineWidth = 4; ctx.setLineDash([6, 4]); }
            else             { ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = 1.5; }
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
        });
        ctx.setLineDash([]);
    }

    drawMacroblockStructure(ctx) {
        if (!this.graphData.macroblock_structure) return;
        const structure = this.graphData.macroblock_structure;

        // Draw macroblock nodes first
        structure.macroblocks.forEach(macro => {
            const size = 10;
            ctx.fillStyle = '#ff6b6b';
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.fillRect(macro.x - size, macro.y - size, size * 2, size * 2);
            ctx.strokeRect(macro.x - size, macro.y - size, size * 2, size * 2);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 10px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(`M${macro.id}`, macro.x, macro.y);
        });

        // Draw macroblock relationship links
        structure.structure_links.forEach(link => {
            const macro = structure.macroblocks.find(m => m.id === link.macro_id);
            const zone  = structure.final_zones.find(z => z.id === link.zone_id);
            if (!macro || !zone) return;
            ctx.strokeStyle = 'rgba(92, 184, 92, 0.7)';
            ctx.lineWidth   = 2;
            ctx.setLineDash([6, 4]);
            ctx.beginPath();
            ctx.moveTo(macro.x, macro.y);
            ctx.lineTo(zone.x, zone.y);
            ctx.stroke();
        });
        ctx.setLineDash([]);

        // Draw final zone structure markers
        structure.final_zones.forEach(zone => {
            const size = 8;
            if (zone.type === 'subdivided') {
                ctx.fillStyle = '#45b7d1';
                ctx.beginPath();
                ctx.moveTo(zone.x, zone.y - size);
                ctx.lineTo(zone.x - size, zone.y + size);
                ctx.lineTo(zone.x + size, zone.y + size);
                ctx.closePath();
                ctx.fill();
            } else if (zone.type === 'combined') {
                ctx.fillStyle = '#96ceb4';
                ctx.beginPath();
                ctx.moveTo(zone.x, zone.y - size);
                ctx.lineTo(zone.x - size, zone.y);
                ctx.lineTo(zone.x, zone.y + size);
                ctx.lineTo(zone.x + size, zone.y);
                ctx.closePath();
                ctx.fill();
            } else {
                ctx.fillStyle = '#ff6b6b';
                ctx.beginPath();
                ctx.arc(zone.x, zone.y, size, 0, 2 * Math.PI);
                ctx.fill();
            }
            ctx.fillStyle = '#ffffff';
            ctx.font = '9px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            if (zone.parent_macro !== undefined && zone.type !== 'macroblock') {
                ctx.fillText(`M${zone.parent_macro}`, zone.x, zone.y + size + 2);
            }
        });
    }

    drawNodes(ctx, p33, p66, min, max) {
        this.graphData.nodes.forEach(node => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.size, 0, 2 * Math.PI);
            ctx.fillStyle   = this.nodeColor(node, p33, p66);
            ctx.fill();
            ctx.strokeStyle = (this.selectedNodes.start === node.id ||
                               this.selectedNodes.destination === node.id) ? '#fff' : '#2c3e50';
            ctx.lineWidth   = (this.selectedNodes.start === node.id ||
                               this.selectedNodes.destination === node.id) ? 3 : 1.5;
            ctx.stroke();
            const rel = max > min ? ((node.congestion - min) / (max - min)).toFixed(2) : '0.00';
            ctx.fillStyle    = '#fff';
            ctx.font         = 'bold 11px Arial';
            ctx.textAlign    = 'center';
            ctx.textBaseline = 'middle';
            const label = node.macroblock_id !== undefined ? `M${node.macroblock_id}` : `Z${node.id}`;
            ctx.fillText(label, node.x, node.y - 4);
            ctx.font = '9px Arial';
            ctx.fillText(rel, node.x, node.y + 7);
        });
    }

    edgeOnPath(path, src, dst) {
        for (let i = 0; i < path.length - 1; i++) {
            if ((path[i] === src && path[i+1] === dst) ||
                (path[i] === dst && path[i+1] === src)) return true;
        }
        return false;
    }

    drawLegend(ctx, p33, p66) {
        const lx = 15, ly = 15;
        const fmt = v => v !== undefined ? v.toFixed(3) : '—';
        ctx.fillStyle = 'rgba(255,255,255,0.92)';
        ctx.fillRect(lx - 5, ly - 5, 215, 115);
        ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1;
        ctx.strokeRect(lx - 5, ly - 5, 215, 115);
        ctx.fillStyle = '#2c3e50'; ctx.font = 'bold 13px Arial';
        ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        ctx.fillText('Congestion Level:', lx, ly + 10);
        [
            { color: '#2ecc71', label: `Low  (< ${fmt(p33)})` },
            { color: '#f39c12', label: `Med  (${fmt(p33)}–${fmt(p66)})` },
            { color: '#e74c3c', label: `High (> ${fmt(p66)})` },
            { color: '#3498db', label: 'Source node' },
            { color: '#9b59b6', label: 'Destination node' },
        ].forEach((l, i) => {
            const y = ly + 28 + i * 17;
            ctx.beginPath(); ctx.arc(lx + 7, y, 6, 0, 2 * Math.PI);
            ctx.fillStyle = l.color; ctx.fill();
            ctx.fillStyle = '#2c3e50'; ctx.font = '11px Arial';
            ctx.textBaseline = 'middle';
            ctx.fillText(l.label, lx + 18, y);
        });
    }

    drawGraph() {
        const canvas = document.getElementById('trafficGraph');
        if (!canvas || !this.graphData) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const { p33, p66, min, max } = this.getPercentileThresholds();
        this.drawBackground(ctx, canvas);
        this.drawEdges(ctx);
        this.drawMacroblockStructure(ctx);
        this.drawNodes(ctx, p33, p66, min, max);
        this.drawLegend(ctx, p33, p66);
    }

    drawRoutingGraph(gnnRoute = null, baselineRoute = null) {
        const canvas = document.getElementById('routingGraph');
        if (!canvas || !this.graphData) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const { p33, p66, min, max } = this.getPercentileThresholds();
        this.drawBackground(ctx, canvas);
        this.drawEdges(ctx, gnnRoute, baselineRoute);
        this.drawMacroblockStructure(ctx);
        this.drawNodes(ctx, p33, p66, min, max);
        this.drawLegend(ctx, p33, p66);

        if (gnnRoute || baselineRoute) {
            const rx = canvas.width - 195, ry = 15;
            ctx.fillStyle = 'rgba(255,255,255,0.92)';
            ctx.fillRect(rx - 5, ry - 5, 185, 60);
            ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1;
            ctx.strokeRect(rx - 5, ry - 5, 185, 60);
            ctx.font = 'bold 11px Arial'; ctx.textAlign = 'left';
            if (gnnRoute) {
                ctx.strokeStyle = '#2ecc71'; ctx.lineWidth = 4; ctx.setLineDash([]);
                ctx.beginPath(); ctx.moveTo(rx, ry + 15); ctx.lineTo(rx + 30, ry + 15); ctx.stroke();
                ctx.fillStyle = '#2c3e50'; ctx.textBaseline = 'middle';
                ctx.fillText('GAT route', rx + 36, ry + 15);
            }
            if (baselineRoute) {
                ctx.strokeStyle = '#e74c3c'; ctx.lineWidth = 3; ctx.setLineDash([5, 4]);
                ctx.beginPath(); ctx.moveTo(rx, ry + 40); ctx.lineTo(rx + 30, ry + 40); ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = '#2c3e50'; ctx.textBaseline = 'middle';
                ctx.fillText('Baseline route', rx + 36, ry + 40);
            }
        }
    }

    // ── Node selection ──────────────────────────────────────────────────────────
    handleCanvasClick(event) {
        if (!this.graphData) return;
        const canvas = document.getElementById('routingGraph');
        const rect   = canvas.getBoundingClientRect();
        const scaleX = canvas.width  / rect.width;
        const scaleY = canvas.height / rect.height;
        const x      = (event.clientX - rect.left) * scaleX;
        const y      = (event.clientY - rect.top)  * scaleY;

        const clicked = this.graphData.nodes.find(n =>
            Math.sqrt((x - n.x) ** 2 + (y - n.y) ** 2) <= n.size + 5
        );

        if (clicked) {
            if (!this.selectedNodes.start)                                           this.selectedNodes.start = clicked.id;
            else if (!this.selectedNodes.destination && clicked.id !== this.selectedNodes.start) this.selectedNodes.destination = clicked.id;
            else if (this.selectedNodes.start === clicked.id)                        this.selectedNodes.start = null;
            else if (this.selectedNodes.destination === clicked.id)                  this.selectedNodes.destination = null;
            this.updateNodeSelection();
            this.drawRoutingGraph();
        }
    }

    updateNodeSelection() {
        document.getElementById('startNode').textContent =
            this.selectedNodes.start !== null ? `Zone ${this.selectedNodes.start}` : 'None';
        document.getElementById('destNode').textContent =
            this.selectedNodes.destination !== null ? `Zone ${this.selectedNodes.destination}` : 'None';
        document.getElementById('calculateRoutesBtn').disabled =
            !(this.selectedNodes.start !== null && this.selectedNodes.destination !== null);
    }

    clearSelection() {
        this.selectedNodes = { start: null, destination: null };
        this.updateNodeSelection();
        this.drawRoutingGraph();
        document.getElementById('gcnRoute').textContent      = '';
        document.getElementById('baselineRoute').textContent = '';
    }

    // ── Routing ─────────────────────────────────────────────────────────────────
    calculateRoutes() {
        const { start, destination } = this.selectedNodes;
        if (start === null || destination === null) return;
        const gnnRoute      = this.dijkstra(this.graphData.adj_gnn,     start, destination);
        const baselineRoute = this.dijkstra(this.graphData.adj_baseline, start, destination);
        this.displayRoutes(gnnRoute, baselineRoute);
        this.drawRoutingGraph(gnnRoute, baselineRoute);
    }

    dijkstra(adj, start, end) {
        const distances = {}, previous = {};
        const unvisited = new Set();
        this.graphData.nodes.forEach(n => {
            distances[n.id] = n.id === start ? 0 : Infinity;
            previous[n.id]  = null;
            unvisited.add(n.id);
        });
        while (unvisited.size > 0) {
            let current = null, minDist = Infinity;
            for (const id of unvisited) {
                if (distances[id] < minDist) { minDist = distances[id]; current = id; }
            }
            if (current === null || distances[current] === Infinity || current === end) break;
            unvisited.delete(current);
            (adj[String(current)] || []).forEach(({ node: nb, weight }) => {
                if (unvisited.has(nb)) {
                    const alt = distances[current] + weight;
                    if (alt < distances[nb]) { distances[nb] = alt; previous[nb] = current; }
                }
            });
        }
        const path = [];
        let cur = end;
        while (cur !== null && cur !== undefined) { path.unshift(cur); cur = previous[cur]; }
        return (path.length > 1 && path[0] === start) ? path : null;
    }

    displayRoutes(gnnRoute, baselineRoute) {
        const avgCong = r => r
            ? r.reduce((s, id) => s + (this.graphData.nodes.find(n => n.id === id)?.congestion || 0), 0) / r.length
            : null;
        const fmt = r => {
            if (!r || !r.length) return 'No route found';
            return `Path: ${r.map(id => `Z${id}`).join(' → ')}\nHops: ${r.length - 1}\nAvg Congestion: ${avgCong(r).toFixed(4)}`;
        };
        let gnnText = fmt(gnnRoute);
        if (gnnRoute && baselineRoute) {
            const gc = avgCong(gnnRoute), bc = avgCong(baselineRoute);
            if (JSON.stringify(gnnRoute) !== JSON.stringify(baselineRoute)) {
                gnnText += gc < bc
                    ? `\n\n✓ ${((bc - gc) / bc * 100).toFixed(1)}% less congestion than baseline`
                    : '\n\nDifferent path, similar congestion';
            } else { gnnText += '\n\nSame route as baseline'; }
        }
        document.getElementById('gcnRoute').textContent      = gnnText;
        document.getElementById('baselineRoute').textContent = fmt(baselineRoute);
    }

    // ── Utilities ───────────────────────────────────────────────────────────────
    showStatus(message, type) {
        const div = document.getElementById('uploadStatus');
        div.textContent = message; div.className = type;
    }
    showProgressBar(show) {
        const bar = document.getElementById('progressBar');
        if (show) bar.classList.remove('hidden'); else bar.classList.add('hidden');
    }
    updateProgressBar(pct) {
        document.getElementById('progressFill').style.width = pct + '%';
        document.getElementById('progressText').textContent = `Processing... ${Math.round(pct)}%`;
    }
}

document.addEventListener('DOMContentLoaded', () => { window.demo = new TrafficDemo(); });
// Traffic Analysis Demo JavaScript
class TrafficDemo {
    constructor() {
        this.currentStep = 0;
        this.selectedNodes = { start: null, destination: null };
        this.graphData = null;
        this.videoFile = null;
        this.syncedVideos = { original: null, yolo: null };
        this.initializeEventListeners();
        this.initializeGraph();
        this.setupSyncedVideoControls();
    }

    initializeEventListeners() {
        // File upload
        document.getElementById('processBtn').addEventListener('click', () => this.processVideo());
        document.getElementById('videoInput').addEventListener('change', (e) => this.handleFileSelect(e));

        // Navigation buttons
        document.getElementById('nextStepBtn').addEventListener('click', () => this.showGraphAnalysis());
        document.getElementById('routingBtn').addEventListener('click', () => this.showRoutingDemo());

        // Routing
        document.getElementById('calculateRoutesBtn').addEventListener('click', () => this.calculateRoutes());
        document.getElementById('clearSelectionBtn').addEventListener('click', () => this.clearSelection());

        // Canvas click for node selection
        document.getElementById('trafficGraph').addEventListener('click', (e) => this.handleCanvasClick(e));
        document.getElementById('routingGraph').addEventListener('click', (e) => this.handleCanvasClick(e, true));
    }

    handleFileSelect(event) {
        const files = event.target.files;
        if (files.length > 0) {
            this.videoFile = files[0];
            this.showStatus(`File selected: ${this.videoFile.name}`, 'success');
        }
    }

    async processVideo() {
        if (!this.videoFile) {
            this.showStatus('Please select a video file first.', 'error');
            return;
        }

        this.showStatus('Processing video with YOLO... This may take a moment.', 'info');
        this.showProgressBar(true);

        // Simulate processing with simulated progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 100) progress = 100;
            this.updateProgressBar(progress);
        }, 300);

        // Create video URLs from the uploaded file
        setTimeout(() => {
            clearInterval(progressInterval);
            this.updateProgressBar(100);
            this.loadAndDisplayVideos();
            setTimeout(() => {
                this.showVideoComparison();
                this.showProgressBar(false);
                this.showStatus('Video processed successfully!', 'success');
            }, 500);
        }, 2000);
    }

    loadAndDisplayVideos() {
        // Create object URLs for the video
        const videoUrl = URL.createObjectURL(this.videoFile);
        
        const originalVideo = document.getElementById('originalVideo');
        const yoloVideo = document.getElementById('yoloVideo');
        
        originalVideo.src = videoUrl;
        // yoloVideo would be the processed video from backend with flask
        // Currently using the original video
        yoloVideo.src = videoUrl;
        
        this.syncedVideos.original = originalVideo;
        this.syncedVideos.yolo = yoloVideo;
    }

    setupSyncedVideoControls() {
        document.getElementById('syncPlayBtn').addEventListener('click', () => this.syncPlayVideos());
        document.getElementById('syncPauseBtn').addEventListener('click', () => this.syncPauseVideos());
        document.getElementById('syncResetBtn').addEventListener('click', () => this.syncResetVideos());
    }

    syncPlayVideos() {
        if (this.syncedVideos.original && this.syncedVideos.yolo) {
            this.syncedVideos.original.play();
            this.syncedVideos.yolo.play();
        }
    }

    syncPauseVideos() {
        if (this.syncedVideos.original && this.syncedVideos.yolo) {
            this.syncedVideos.original.pause();
            this.syncedVideos.yolo.pause();
        }
    }

    syncResetVideos() {
        if (this.syncedVideos.original && this.syncedVideos.yolo) {
            this.syncedVideos.original.currentTime = 0;
            this.syncedVideos.yolo.currentTime = 0;
            this.syncedVideos.original.pause();
            this.syncedVideos.yolo.pause();
        }
    }

    showVideoComparison() {
        this.hideAllSections();
        document.getElementById('video-comparison').classList.remove('hidden');
    }

    showGraphAnalysis() {
        this.hideAllSections();
        document.getElementById('graph-analysis').classList.remove('hidden');
        // Draw the graph when shown
        setTimeout(() => this.drawGraph(), 100);
    }

    showRoutingDemo() {
        this.hideAllSections();
        document.getElementById('routing-demo').classList.remove('hidden');
        // Draw routing graph
        setTimeout(() => this.drawRoutingGraph(), 100);
    }

    hideAllSections() {
        document.getElementById('video-comparison').classList.add('hidden');
        document.getElementById('graph-analysis').classList.add('hidden');
        document.getElementById('routing-demo').classList.add('hidden');
    }

    initializeGraph() {
        // Generate demo graph data
        this.graphData = this.generateSampleGraphData();
    }

    generateSampleGraphData() {
        // Create a 4x5 grid of traffic zones
        const nodes = [];
        const edges = [];
        const nodePositions = {};
        
        const rows = 4;
        const cols = 5;
        const spacing = 120;
        const startX = 100;
        const startY = 100;

        // Generate nodes in grid layout
        let nodeId = 0;
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const x = startX + j * spacing;
                const y = startY + i * spacing;
                const traffic = Math.random() * 5; // 0-5 traffic level
                
                nodes.push({
                    id: nodeId,
                    x: x,
                    y: y,
                    size: 15,
                    traffic: traffic,
                    label: `Z${nodeId}`
                });
                
                nodePositions[nodeId] = { x, y };
                nodeId++;
            }
        }

        // Generate edges (connect adjacent nodes and some random ones)
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const currentId = i * cols + j;
                
                // Connect to right neighbor
                if (j < cols - 1) {
                    edges.push([currentId, i * cols + (j + 1)]);
                }
                
                // Connect to bottom neighbor
                if (i < rows - 1) {
                    edges.push([currentId, (i + 1) * cols + j]);
                }
            }
        }

        return {
            nodes: nodes,
            edges: edges,
            nodeCount: nodes.length,
            edgeCount: edges.length
        };
    }

    drawGraph() {
        const canvas = document.getElementById('trafficGraph');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw edges first
        ctx.strokeStyle = '#95a5a6';
        ctx.lineWidth = 2;
        this.graphData.edges.forEach(([from, to]) => {
            const nodeFrom = this.graphData.nodes[from];
            const nodeTo = this.graphData.nodes[to];
            if (nodeFrom && nodeTo) {
                ctx.beginPath();
                ctx.moveTo(nodeFrom.x, nodeFrom.y);
                ctx.lineTo(nodeTo.x, nodeTo.y);
                ctx.stroke();
            }
        });

        // Draw nodes
        this.graphData.nodes.forEach(node => {
            // Node circle with gradient based on traffic
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.size, 0, 2 * Math.PI);

            // Color based on traffic level (green to red)
            const trafficRatio = Math.min(node.traffic / 5, 1); // Cap at 1
            const red = Math.floor(46 + (230 * trafficRatio)); // 46 to 276 (capped at 255)
            const green = Math.floor(204 - (80 * trafficRatio)); // 204 to 124
            const blue = 113; // constant
            
            ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
            ctx.fill();

            // Node border, highlight if selected
            ctx.strokeStyle = this.selectedNodes.start === node.id || this.selectedNodes.destination === node.id ? '#e74c3c' : '#2c3e50';
            ctx.lineWidth = this.selectedNodes.start === node.id || this.selectedNodes.destination === node.id ? 3 : 2;
            ctx.stroke();

            // Node label
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(node.id.toString(), node.x, node.y);
        });

        // Draw legend
        this.drawLegend(ctx);
    }

    drawRoutingGraph() {
        const canvas = document.getElementById('routingGraph');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw edges with width proportional to traffic
        this.graphData.edges.forEach(([from, to]) => {
            const nodeFrom = this.graphData.nodes[from];
            const nodeTo = this.graphData.nodes[to];
            if (nodeFrom && nodeTo) {
                ctx.strokeStyle = '#3498db';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(nodeFrom.x, nodeFrom.y);
                ctx.lineTo(nodeTo.x, nodeTo.y);
                ctx.stroke();
            }
        });

        // Draw nodes
        this.graphData.nodes.forEach(node => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.size, 0, 2 * Math.PI);

            const trafficRatio = Math.min(node.traffic / 5, 1);
            const red = Math.floor(46 + (230 * trafficRatio));
            const green = Math.floor(204 - (80 * trafficRatio));
            const blue = 113;
            
            ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
            ctx.fill();

            ctx.strokeStyle = this.selectedNodes.start === node.id ? '#27ae60' : (this.selectedNodes.destination === node.id ? '#e74c3c' : '#2c3e50');
            ctx.lineWidth = this.selectedNodes.start === node.id || this.selectedNodes.destination === node.id ? 3 : 2;
            ctx.stroke();

            ctx.fillStyle = '#fff';
            ctx.font = 'bold 12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(node.id.toString(), node.x, node.y);
        });

        // Draw legend
        this.drawLegend(ctx);
    }

    drawLegend(ctx) {
        const legendX = 50;
        const legendY = 50;

        ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
        ctx.fillRect(legendX - 10, legendY - 10, 220, 100);

        ctx.fillStyle = '#2c3e50';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Congestion Level:', legendX, legendY + 10);

        // Color gradient legend
        const gradient = ctx.createLinearGradient(legendX, legendY + 20, legendX + 100, legendY + 20);
        gradient.addColorStop(0, 'rgb(46, 204, 113)');     // Green - low
        gradient.addColorStop(0.5, 'rgb(241, 196, 15)');   // Yellow - medium
        gradient.addColorStop(1, 'rgb(231, 76, 60)');      // Red - high
        ctx.fillStyle = gradient;
        ctx.fillRect(legendX, legendY + 20, 100, 20);

        ctx.fillStyle = '#2c3e50';
        ctx.font = '12px Arial';
        ctx.fillText('Low', legendX, legendY + 50);
        ctx.fillText('High', legendX + 80, legendY + 50);
    }

    handleCanvasClick(event, isRoutingCanvas = false) {
        const canvasId = isRoutingCanvas ? 'routingGraph' : 'trafficGraph';
        const canvas = document.getElementById(canvasId);
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // Find clicked node
        const clickedNode = this.graphData.nodes.find(node => {
            const distance = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2);
            return distance <= node.size + 5; // Add some margin for easier clicking
        });

        if (clickedNode) {
            if (!this.selectedNodes.start) {
                this.selectedNodes.start = clickedNode.id;
            } else if (!this.selectedNodes.destination && this.selectedNodes.start !== clickedNode.id) {
                this.selectedNodes.destination = clickedNode.id;
            } else if (this.selectedNodes.start === clickedNode.id) {
                this.selectedNodes.start = null;
            } else if (this.selectedNodes.destination === clickedNode.id) {
                this.selectedNodes.destination = null;
            }

            this.updateNodeSelection();
            if (isRoutingCanvas) {
                this.drawRoutingGraph();
            } else {
                this.drawGraph();
            }
        }
    }

    updateNodeSelection() {
        document.getElementById('startNode').textContent = this.selectedNodes.start !== null ? `Node ${this.selectedNodes.start}` : 'None';
        document.getElementById('destNode').textContent = this.selectedNodes.destination !== null ? `Node ${this.selectedNodes.destination}` : 'None';

        const calculateBtn = document.getElementById('calculateRoutesBtn');
        calculateBtn.disabled = !(this.selectedNodes.start !== null && this.selectedNodes.destination !== null);
    }

    clearSelection() {
        this.selectedNodes = { start: null, destination: null };
        this.updateNodeSelection();
        this.drawRoutingGraph();
        document.getElementById('gcnRoute').textContent = '';
        document.getElementById('baselineRoute').textContent = '';
    }

    calculateRoutes() {
        const start = this.selectedNodes.start;
        const dest = this.selectedNodes.destination;

        if (start === null || dest === null) {
            this.showStatus('Please select both start and destination nodes', 'error');
            return;
        }

        // Calculate both routes
        const gnnRoute = this.findGCNRoute(start, dest);
        const baselineRoute = this.findBaselineRoute(start, dest);

        this.displayRoutes(gnnRoute, baselineRoute);
    }

    findGCNRoute(start, dest) {
        // GCN-aware route: considers traffic/congestion weights
        // Use Dijkstra's with congestion as weight
        return this.dijkstra(start, dest, true);
    }

    findBaselineRoute(start, dest) {
        // Baseline route: uniform weights (shortest path)
        return this.dijkstra(start, dest, false);
    }

    dijkstra(start, end, useTraffic = false) {
        const distances = {};
        const previous = {};
        const unvisited = new Set();

        // Initialize distances
        this.graphData.nodes.forEach(node => {
            distances[node.id] = node.id === start ? 0 : Infinity;
            previous[node.id] = null;
            unvisited.add(node.id);
        });

        while (unvisited.size > 0) {
            // Find unvisited node with minimum distance
            let current = null;
            let minDist = Infinity;
            
            for (let nodeId of unvisited) {
                if (distances[nodeId] < minDist) {
                    minDist = distances[nodeId];
                    current = nodeId;
                }
            }

            if (current === null || distances[current] === Infinity) break;
            if (current === end) break;

            unvisited.delete(current);

            // Check all neighbors
            this.graphData.edges.forEach(([from, to]) => {
                let neighbor = null;
                if (from === current) neighbor = to;
                else if (to === current) neighbor = from;

                if (neighbor !== null && unvisited.has(neighbor)) {
                    // Calculate weight
                    let weight = 1;
                    if (useTraffic) {
                        const neighborNode = this.graphData.nodes[neighbor];
                        weight = 1 + (neighborNode.traffic / 5); // Weight based on congestion
                    }

                    const altDist = distances[current] + weight;
                    if (altDist < distances[neighbor]) {
                        distances[neighbor] = altDist;
                        previous[neighbor] = current;
                    }
                }
            });
        }

        // Reconstruct path
        const path = [];
        let current = end;
        while (current !== null) {
            path.unshift(current);
            current = previous[current];
        }

        // Return path only if it exists
        return path.length > 1 || (path.length === 1 && path[0] === start) ? path : [];
    }

    displayRoutes(gcnRoute, baselineRoute) {
        const formatRoute = (route) => {
            if (!route || route.length === 0) return 'No route found';

            const path = route.map(id => `Z${id}`).join(' → ');
            const totalTraffic = route.reduce((sum, nodeId) => {
                const node = this.graphData.nodes.find(n => n.id === nodeId);
                return sum + (node ? node.traffic : 0);
            }, 0);

            return `Path: ${path}\nTotal Congestion: ${totalTraffic.toFixed(1)}\nHops: ${route.length - 1}`;
        };

        document.getElementById('gcnRoute').textContent = formatRoute(gcnRoute);
        document.getElementById('baselineRoute').textContent = formatRoute(baselineRoute);
        
        // Highlight the routes on the canvas
        this.highlightRoute(gcnRoute, baselineRoute);
    }

    highlightRoute(gnnRoute, baselineRoute) {
        const canvas = document.getElementById('routingGraph');
        const ctx = canvas.getContext('2d');

        // Redraw everything
        this.drawRoutingGraph();

        // Highlight GNN route in green
        if (gnnRoute && gnnRoute.length > 1) {
            ctx.strokeStyle = 'rgba(46, 204, 113, 0.8)';
            ctx.lineWidth = 4;
            for (let i = 0; i < gnnRoute.length - 1; i++) {
                const from = this.graphData.nodes[gnnRoute[i]];
                const to = this.graphData.nodes[gnnRoute[i + 1]];
                ctx.beginPath();
                ctx.moveTo(from.x, from.y);
                ctx.lineTo(to.x, to.y);
                ctx.stroke();
            }
        }

        // Highlight baseline route in blue
        if (baselineRoute && baselineRoute.length > 1) {
            ctx.strokeStyle = 'rgba(52, 152, 219, 0.8)';
            ctx.lineWidth = 4;
            ctx.setLineDash([5, 5]);
            for (let i = 0; i < baselineRoute.length - 1; i++) {
                const from = this.graphData.nodes[baselineRoute[i]];
                const to = this.graphData.nodes[baselineRoute[i + 1]];
                ctx.beginPath();
                ctx.moveTo(from.x, from.y);
                ctx.lineTo(to.x, to.y);
                ctx.stroke();
            }
            ctx.setLineDash([]);
        }
    }

    showStatus(message, type) {
        const statusDiv = document.getElementById('uploadStatus');
        statusDiv.textContent = message;
        statusDiv.className = type;
    }

    showProgressBar(show) {
        const progressBar = document.getElementById('progressBar');
        if (show) {
            progressBar.classList.remove('hidden');
        } else {
            progressBar.classList.add('hidden');
        }
    }

    updateProgressBar(percentage) {
        const fill = document.getElementById('progressFill');
        const text = document.getElementById('progressText');
        fill.style.width = percentage + '%';
        text.textContent = `Processing... ${Math.round(percentage)}%`;
    }
}

// Initialize demo when page loads
document.addEventListener('DOMContentLoaded', () => {
    new TrafficDemo();
});
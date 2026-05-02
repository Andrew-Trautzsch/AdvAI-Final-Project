// Traffic Analysis Demo JavaScript
class TrafficDemo {
    constructor() {
        this.currentStep = 0;
        this.selectedNodes = { start: null, destination: null };
        this.graphData = null;
        this.initializeEventListeners();
        this.initializeGraph();
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

        // Canvas click for node selection
        document.getElementById('trafficGraph').addEventListener('click', (e) => this.handleCanvasClick(e));
    }

    handleFileSelect(event) {
        const files = event.target.files;
        if (files.length > 0) {
            this.showStatus('Files selected successfully!', 'success');
        }
    }

    async processVideo() {
        const videoInput = document.getElementById('videoInput');
        const files = videoInput.files;

        if (files.length === 0) {
            this.showStatus('Please select video files first.', 'error');
            return;
        }

        this.showStatus('Processing video with YOLO... This may take a moment.', 'info');

        // Simulate processing delay
        setTimeout(() => {
            this.showVideoComparison();
        }, 2000);
    }

    showVideoComparison() {

    }

    showGraphAnalysis() {

    }

    showRoutingDemo() {

    }

    initializeGraph() {

    }

    drawGraph() {
        const canvas = document.getElementById('trafficGraph');
        const ctx = canvas.getContext('2d');

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw edges
        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 2;
        this.graphData.edges.forEach(([from, to]) => {
            const nodeFrom = this.graphData.nodes[from];
            const nodeTo = this.graphData.nodes[to];
            ctx.beginPath();
            ctx.moveTo(nodeFrom.x, nodeFrom.y);
            ctx.lineTo(nodeTo.x, nodeTo.y);
            ctx.stroke();
        });

        // Draw nodes
        this.graphData.nodes.forEach(node => {
            // Node circle
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.size, 0, 2 * Math.PI);

            // Color based on traffic level
            const trafficRatio = node.traffic / 5; // Assuming max traffic is 5
            const green = Math.floor(255 * (1 - trafficRatio));
            const red = Math.floor(255 * trafficRatio);
            ctx.fillStyle = `rgb(${red}, ${green}, 100)`;
            ctx.fill();

            // Node border
            ctx.strokeStyle = this.selectedNodes.start === node.id || this.selectedNodes.destination === node.id ? '#e74c3c' : '#2c3e50';
            ctx.lineWidth = this.selectedNodes.start === node.id || this.selectedNodes.destination === node.id ? 3 : 2;
            ctx.stroke();

            // Node label
            ctx.fillStyle = '#fff';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(node.id.toString(), node.x, node.y + 4);
        });

        // Legend
        this.drawLegend(ctx);
    }

    drawLegend(ctx) {
        const legendX = 50;
        const legendY = 50;

        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.fillRect(legendX - 10, legendY - 10, 200, 80);

        ctx.fillStyle = '#2c3e50';
        ctx.font = '14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Traffic Intensity:', legendX, legendY + 10);

        // Color gradient
        const gradient = ctx.createLinearGradient(legendX, legendY + 20, legendX + 100, legendY + 20);
        gradient.addColorStop(0, 'rgb(0, 255, 100)');
        gradient.addColorStop(1, 'rgb(255, 0, 100)');
        ctx.fillStyle = gradient;
        ctx.fillRect(legendX, legendY + 20, 100, 20);

        ctx.fillStyle = '#2c3e50';
        ctx.font = '12px Arial';
        ctx.fillText('Low', legendX, legendY + 50);
        ctx.fillText('High', legendX + 80, legendY + 50);
    }

    handleCanvasClick(event) {
        const canvas = document.getElementById('trafficGraph');
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // Find clicked node
        const clickedNode = this.graphData.nodes.find(node => {
            const distance = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2);
            return distance <= node.size;
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
            this.drawGraph();
        }
    }

    updateNodeSelection() {
        document.getElementById('startNode').textContent = this.selectedNodes.start ?? 'None';
        document.getElementById('destNode').textContent = this.selectedNodes.destination ?? 'None';

        const calculateBtn = document.getElementById('calculateRoutesBtn');
        calculateBtn.disabled = !(this.selectedNodes.start !== null && this.selectedNodes.destination !== null);
    }

    calculateRoutes() {

    }

    findGCNRoute(start, dest) {

    }

    findBaselineRoute(start, dest) {

    }

    displayRoutes(gcnRoute, baselineRoute) {
        const formatRoute = (route) => {
            if (!route || route.length === 0) return 'No route found';

            const path = route.join(' → ');
            const totalTraffic = route.reduce((sum, nodeId) => {
                return sum + this.graphData.nodes[nodeId].traffic;
            }, 0);

            return `Path: ${path}\nTotal Traffic: ${totalTraffic.toFixed(1)}\nSteps: ${route.length - 1}`;
        };

        document.getElementById('gcnRoute').textContent = formatRoute(gcnRoute);
        document.getElementById('baselineRoute').textContent = formatRoute(baselineRoute);
    }

    showStatus(message, type) {
        const statusDiv = document.getElementById('uploadStatus');
        statusDiv.textContent = message;
        statusDiv.className = type;
    }
}

// Initialize demo when page loads
document.addEventListener('DOMContentLoaded', () => {
    new TrafficDemo();
});
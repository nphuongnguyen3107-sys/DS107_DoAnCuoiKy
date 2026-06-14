/* static/js/app.js */

document.addEventListener("DOMContentLoaded", () => {
    // ----------------- KHỞI TẠO CÁC BIẾN & ĐỐI TƯỢNG CHART -----------------
    let gaugeChart = null;
    let shapChart = null;
    let sampleData = null; // Lưu các mẫu dữ liệu từ API
    let currentStrainContext = null; // Lưu kết quả chủng vi khuẩn phục vụ chat
    let chatHistory = []; // Lưu lịch sử trò chuyện

    // DOM Elements
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const jsonInput = document.getElementById("json-input");
    
    // Model Status Info
    const modelNameEl = document.getElementById("loaded-model-name");
    const modelThresholdEl = document.getElementById("model-threshold");
    const modelFeaturesCountEl = document.getElementById("model-features-count");

    // Single Diagnostic DOMs
    const btnPredictSingle = document.getElementById("btn-predict-single");
    const loadSusceptibleBtn = document.getElementById("load-susceptible-btn");
    const loadResistantBtn = document.getElementById("load-resistant-btn");
    const singleEmptyState = document.getElementById("single-empty-state");
    const singleResultDetails = document.getElementById("single-result-details");
    const predictionTag = document.getElementById("prediction-tag");
    const probPercentage = document.getElementById("prob-percentage");
    const shapCard = document.getElementById("shap-card");
    const aiCard = document.getElementById("ai-card");
    const aiReportBox = document.getElementById("ai-report-box");
    const aiChatHistory = document.getElementById("ai-chat-history");
    const aiChatInput = document.getElementById("ai-chat-input");
    const btnSendChat = document.getElementById("btn-send-chat");
    
    // History & SHAP Download Elements
    const patientIdInput = document.getElementById("patient-id-input");
    const btnDownloadShap = document.getElementById("btn-download-shap");
    const historyTableBody = document.getElementById("history-table-body");
    const btnClearHistory = document.getElementById("btn-clear-history");

    // Batch Diagnostic DOMs
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("batch-file-input");
    const uploadActionsPanel = document.getElementById("upload-actions-panel");
    const selectedFileName = document.getElementById("selected-file-name");
    const btnPredictBatch = document.getElementById("btn-predict-batch");
    const batchResultsCard = document.getElementById("batch-results-card");
    const batchPreviewTable = document.getElementById("batch-preview-table").querySelector("tbody");
    const btnDownloadResults = document.getElementById("btn-download-results");

    // Helper function to switch tabs programmatically
    function switchTab(tabId) {
        navButtons.forEach(btn => {
            if (btn.getAttribute("data-tab") === tabId) {
                navButtons.forEach(b => b.classList.remove("active"));
                tabPanes.forEach(p => p.classList.remove("active"));
                
                btn.classList.add("active");
                document.getElementById(tabId).classList.add("active");
                
                if (tabId === "tab-history") {
                    loadHistory();
                } else if (tabId === "tab-epidemiology") {
                    loadEpidemiologyStats();
                }
            }
        });
    }

    // ----------------- CHUYỂN TAB ĐIỀU HƯỚNG -----------------
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            navButtons.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetTab = btn.getAttribute("data-tab");
            document.getElementById(targetTab).classList.add("active");
            
            if (targetTab === "tab-history") {
                loadHistory();
            } else if (targetTab === "tab-epidemiology") {
                loadEpidemiologyStats();
            }
        });
    });

    // ----------------- TẢI THÔNG TIN MÔ HÌNH & DỮ LIỆU MẪU -----------------
    async function loadModelInfo() {
        try {
            const res = await fetch("/api/model_info");
            const data = await res.json();
            if (data.status === "success") {
                modelNameEl.textContent = data.model_name;
                modelThresholdEl.textContent = data.threshold;
                modelFeaturesCountEl.textContent = data.features_count;
            } else {
                modelNameEl.textContent = "Không tìm thấy mô hình";
            }
        } catch (err) {
            console.error("Lỗi tải thông tin mô hình:", err);
            modelNameEl.textContent = "Lỗi kết nối";
        }
    }

    async function loadSampleData() {
        try {
            const res = await fetch("/api/get_samples");
            const data = await res.json();
            if (data.status === "success") {
                sampleData = data.samples;
            }
        } catch (err) {
            console.error("Lỗi tải dữ liệu mẫu:", err);
        }
    }

    loadModelInfo();
    loadSampleData();



    let susceptibleIndex = 0;
    let resistantIndex = 0;

    // Sự kiện nút nạp mẫu có sẵn
    loadSusceptibleBtn.addEventListener("click", () => {
        if (sampleData && sampleData.susceptible && sampleData.susceptible.length > 0) {
            const sample = sampleData.susceptible[susceptibleIndex % sampleData.susceptible.length];
            jsonInput.value = JSON.stringify(sample.features, null, 2);
            patientIdInput.value = sample.id;
            susceptibleIndex++;
        } else {
            alert("Không tìm thấy dữ liệu mẫu nhạy cảm. Bạn hãy chạy file run_training.py trước.");
        }
    });

    loadResistantBtn.addEventListener("click", () => {
        if (sampleData && sampleData.resistant && sampleData.resistant.length > 0) {
            const sample = sampleData.resistant[resistantIndex % sampleData.resistant.length];
            jsonInput.value = JSON.stringify(sample.features, null, 2);
            patientIdInput.value = sample.id;
            resistantIndex++;
        } else {
            alert("Không tìm thấy dữ liệu mẫu kháng thuốc. Bạn hãy chạy file run_training.py trước.");
        }
    });

    // ----------------- PHÂN TÍCH ĐƠN LẺ & SHAP -----------------
    btnPredictSingle.addEventListener("click", async () => {
        const rawJson = jsonInput.value.trim();
        if (!rawJson) {
            alert("Vui lòng nhập JSON đặc trưng chủng vi khuẩn hoặc chọn mẫu có sẵn!");
            return;
        }

        let parsedFeatures;
        try {
            parsedFeatures = JSON.parse(rawJson);
        } catch (err) {
            alert("Định dạng JSON không hợp lệ! Hãy kiểm tra lại dấu phẩy hoặc ngoặc kép.");
            return;
        }

        btnPredictSingle.disabled = true;
        btnPredictSingle.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang tính toán...';

        // Khởi động giao diện Trợ lý AI và reset lịch sử chat
        aiCard.classList.remove("hidden");
        aiReportBox.innerHTML = '<div class="ai-loading"><i class="fa-solid fa-circle-notch fa-spin"></i> Đang chuẩn bị phân tích sinh tin học...</div>';
        aiChatHistory.innerHTML = '';
        aiChatInput.value = '';
        aiChatInput.disabled = true;
        btnSendChat.disabled = true;
        chatHistory = [];
        currentStrainContext = null;

        const patientIdVal = patientIdInput.value.trim();
        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    features: parsedFeatures,
                    strain_id: patientIdVal,
                    patient_id: patientIdVal // Cho tính tương thích ngược
                })
            });
            const data = await res.json();
            
            if (data.status === "success") {
                // Tự động điền mã chủng vi khuẩn nếu trống
                patientIdInput.value = data.strain_id || data.patient_id;
                
                const pred = data.prediction;
                const prob = pred.prob_resistant;
                const resultText = pred.prediction;
                const thresholdUsed = pred.threshold_used;

                // Lưu ngữ cảnh phục vụ chat
                currentStrainContext = {
                    prediction: resultText,
                    prob_resistant: prob,
                    top_features: data.shap ? data.shap.top_features : []
                };

                // 1. Hiển thị kết luận
                singleEmptyState.classList.add("hidden");
                singleResultDetails.classList.remove("hidden");
                
                predictionTag.className = "badge-outcome " + resultText.toLowerCase();
                predictionTag.textContent = resultText === "Resistant" ? "KHÁNG CIPROFLOXACIN (Resistant)" : "NHẠY CẢM CIPROFLOXACIN (Susceptible)";

                // 2. Vẽ biểu đồ Gauge đo xác suất kháng thuốc
                probPercentage.textContent = (prob * 100).toFixed(1) + "%";
                drawGaugeChart(prob, thresholdUsed);

                // 3. Vẽ biểu đồ SHAP giải thích mô hình
                if (data.shap && data.shap.top_features) {
                    shapCard.classList.remove("hidden");
                    drawShapChart(data.shap.top_features);
                } else {
                    shapCard.classList.add("hidden");
                }

                // 4. Render Báo cáo Sinh tin học AI (Marked.js)
                if (data.ai_report) {
                    aiReportBox.innerHTML = marked.parse(data.ai_report);
                    
                    // Thêm lời chào đầu tiên từ AI vào khung chat
                    appendChatMessage("bot", "Chào bạn! Tôi đã hoàn thành báo cáo phân tích sinh tin học ban đầu cho chủng vi khuẩn này. Bạn có thể hỏi thêm về cơ chế đề kháng Ciprofloxacin của các đột biến gen được phát hiện hoặc yêu cầu gợi ý hướng xử lý.");
                    
                    // Kích hoạt ô nhập chat
                    aiChatInput.disabled = false;
                    btnSendChat.disabled = false;
                }
            } else {
                alert("Lỗi từ máy chủ: " + data.message);
                aiCard.classList.add("hidden");
            }
        } catch (err) {
            alert("Không thể kết nối đến máy chủ: " + err);
            aiCard.classList.add("hidden");
        } finally {
            btnPredictSingle.disabled = false;
            btnPredictSingle.innerHTML = '<i class="fa-solid fa-play"></i> Tiến hành phân tích & Giải thích SHAP';
        }
    });

    // Hàm vẽ Gauge Chart (Half-Doughnut)
    function drawGaugeChart(probability, threshold) {
        const ctx = document.getElementById("gaugeChart").getContext("2d");
        
        if (gaugeChart) {
            gaugeChart.destroy();
        }

        // Chọn màu sắc chính phụ thuộc kết quả vượt ngưỡng
        const color = probability >= threshold ? '#ef4444' : '#10b981';

        gaugeChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [probability * 100, (1 - probability) * 100],
                    backgroundColor: [color, 'rgba(255, 255, 255, 0.05)'],
                    borderWidth: 0,
                    borderRadius: [10, 0]
                }]
            },
            options: {
                rotation: 270,
                circumference: 180,
                cutout: '80%',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                }
            }
        });
    }

    // Hàm vẽ biểu đồ cột ngang SHAP
    function drawShapChart(topFeatures) {
        const ctx = document.getElementById("shapChart").getContext("2d");
        
        if (shapChart) {
            shapChart.destroy();
        }

        const labels = topFeatures.map(f => f.feature);
        const values = topFeatures.map(f => f.shap_value);
        
        // Màu sắc: Đỏ cho giá trị dương (kháng), Xanh lục cho giá trị âm (nhạy cảm)
        const backgroundColors = values.map(v => v >= 0 ? 'rgba(239, 68, 68, 0.75)' : 'rgba(16, 185, 129, 0.75)');
        const borderColors = values.map(v => v >= 0 ? '#ef4444' : '#10b981');

        shapChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Tác động SHAP (Đẩy sang Kháng Ciprofloxacin nếu dương)',
                    data: values,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y', // Biến thành cột ngang
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `SHAP value: ${context.raw.toFixed(4)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#f8fafc', font: { weight: '500' } }
                    }
                }
            }
        });
    }

    // ----------------- TƯƠNG TÁC HỘI THOẠI AI CHATBOT -----------------
    function appendChatMessage(role, content) {
        const div = document.createElement("div");
        div.className = `chat-message ${role}`;
        if (role === 'bot') {
            div.innerHTML = content;
        } else {
            div.textContent = content;
        }
        aiChatHistory.appendChild(div);
        aiChatHistory.scrollTop = aiChatHistory.scrollHeight;
        return div;
    }

    async function sendChatMessage() {
        const message = aiChatInput.value.trim();
        if (!message) return;

        // 1. Thêm tin nhắn của người dùng
        appendChatMessage("user", message);
        aiChatInput.value = "";

        // Tạm khóa ô nhập
        aiChatInput.disabled = true;
        btnSendChat.disabled = true;

        // 2. Thêm chỉ báo đang tải (typing indicator)
        const typingIndicator = appendChatMessage("bot", '<i class="fa-solid fa-circle-notch fa-spin"></i> Đang phân tích...');

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: message,
                    history: chatHistory,
                    context: currentStrainContext
                })
            });
            const data = await res.json();

            // Xóa chỉ báo đang tải
            typingIndicator.remove();

            if (data.status === "success") {
                // Render markdown câu trả lời
                appendChatMessage("bot", marked.parse(data.reply));
                
                // Thêm vào lịch sử hội thoại
                chatHistory.push({ role: "user", content: message });
                chatHistory.push({ role: "model", content: data.reply });
            } else {
                appendChatMessage("bot", `<span class="text-danger"><i class="fa-solid fa-circle-exclamation"></i> Lỗi: ${data.message}</span>`);
            }
        } catch (err) {
            typingIndicator.remove();
            appendChatMessage("bot", `<span class="text-danger"><i class="fa-solid fa-circle-exclamation"></i> Lỗi kết nối: ${err}</span>`);
        } finally {
            // Mở khóa nhập
            aiChatInput.disabled = false;
            btnSendChat.disabled = false;
            aiChatInput.focus();
            aiChatHistory.scrollTop = aiChatHistory.scrollHeight;
        }
    }

    btnSendChat.addEventListener("click", sendChatMessage);
    aiChatInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            sendChatMessage();
        }
    });

    // ----------------- CHẨN ĐOÁN HÀNG LOẠT (UPLOAD CSV) -----------------
    // Drag & Drop events
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "#6366f1";
        dropzone.style.backgroundColor = "rgba(99, 102, 241, 0.08)";
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "rgba(99, 102, 241, 0.3)";
        dropzone.style.backgroundColor = "rgba(99, 102, 241, 0.02)";
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "rgba(99, 102, 241, 0.3)";
        dropzone.style.backgroundColor = "rgba(99, 102, 241, 0.02)";
        
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.name.endsWith('.csv')) {
            alert("Chỉ chấp nhận file định dạng CSV!");
            return;
        }
        selectedFileName.textContent = file.name;
        uploadActionsPanel.classList.remove("hidden");
    }

    // Gửi file chẩn đoán hàng loạt
    btnPredictBatch.addEventListener("click", async () => {
        const file = fileInput.files[0] || (fileInput.files.length ? fileInput.files[0] : null);
        if (!file) return;

        btnPredictBatch.disabled = true;
        btnPredictBatch.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang xử lý...';

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/predict_batch", {
                method: "POST",
                body: formData
            });
            const data = await res.json();

            if (data.status === "success") {
                batchResultsCard.classList.remove("hidden");
                btnDownloadResults.href = data.download_url;
                
                // Hiển thị bảng xem trước (preview)
                batchPreviewTable.innerHTML = "";
                data.preview.forEach(row => {
                    const tr = document.createElement("tr");
                    
                    const tdId = document.createElement("td");
                    tdId.textContent = row.Sample_ID;
                    tr.appendChild(tdId);
                    
                    const tdPred = document.createElement("td");
                    const span = document.createElement("span");
                    span.className = "badge " + (row.Prediction === "Resistant" ? "label-resistant" : "label-susceptible");
                    span.textContent = row.Prediction === "Resistant" ? "Kháng Ciprofloxacin (Resistant)" : "Nhạy cảm (Susceptible)";
                    tdPred.appendChild(span);
                    tr.appendChild(tdPred);
                    
                    const tdProb = document.createElement("td");
                    tdProb.innerHTML = `<strong>${(row.Probability_Resistant * 100).toFixed(2)}%</strong>`;
                    tr.appendChild(tdProb);

                    batchPreviewTable.appendChild(tr);
                });
            } else {
                alert("Lỗi chẩn đoán hàng loạt: " + data.message);
            }
        } catch (err) {
            alert("Lỗi kết nối máy chủ: " + err);
        } finally {
            btnPredictBatch.disabled = false;
            btnPredictBatch.innerHTML = '<i class="fa-solid fa-gears"></i> Bắt đầu xử lý hàng loạt';
        }
    });

    // ----------------- TỪ ĐIỂN GEN KHÁNG THUỐC (AMR GENE DICTIONARY) -----------------
    let fullGeneDb = {}; // Lưu trữ toàn bộ dữ liệu gen từ server
    const geneCardsContainer = document.getElementById("gene-cards-container");
    const geneSearchInput = document.getElementById("gene-search-input");

    async function loadGeneDictionary() {
        try {
            const res = await fetch("/api/gene_db");
            const data = await res.json();
            if (data.status === "success") {
                fullGeneDb = data.gene_db;
                renderGeneCards(fullGeneDb);
            } else {
                geneCardsContainer.innerHTML = '<div class="empty-state"><p class="text-danger">Không tải được cơ sở dữ liệu gen.</p></div>';
            }
        } catch (err) {
            console.error("Lỗi tải từ điển gen:", err);
            geneCardsContainer.innerHTML = '<div class="empty-state"><p class="text-danger">Lỗi kết nối máy chủ.</p></div>';
        }
    }

    function renderGeneCards(db) {
        geneCardsContainer.innerHTML = "";
        
        const keys = Object.keys(db);
        if (keys.length === 0) {
            geneCardsContainer.innerHTML = '<div class="empty-state"><p>Không tìm thấy gen kháng thuốc nào khớp.</p></div>';
            return;
        }

        keys.forEach(gene => {
            const desc = db[gene];
            
            // Xác định loại (đột biến hay gen nguyên bản)
            const isMutation = gene.includes("_") || gene.toLowerCase().includes("delta");
            const typeClass = isMutation ? "type-mut" : "type-gene";
            const typeLabel = isMutation ? "Đột Biến" : "Gen Kháng";

            const card = document.createElement("div");
            card.className = "gene-card";

            card.innerHTML = `
                <div class="gene-card-header">
                    <span class="gene-name">${gene}</span>
                    <span class="gene-type-badge ${typeClass}">${typeLabel}</span>
                </div>
                <p class="gene-desc">${desc}</p>
            `;
            geneCardsContainer.appendChild(card);
        });
    }

    // Sự kiện tìm kiếm lọc danh sách gen
    geneSearchInput.addEventListener("input", () => {
        const query = geneSearchInput.value.toLowerCase().trim();
        if (!query) {
            renderGeneCards(fullGeneDb);
            return;
        }

        const filteredDb = {};
        Object.keys(fullGeneDb).forEach(gene => {
            if (gene.toLowerCase().includes(query) || fullGeneDb[gene].toLowerCase().includes(query)) {
                filteredDb[gene] = fullGeneDb[gene];
            }
        });
        renderGeneCards(filteredDb);
    });

    // Sự kiện tải ảnh SHAP (PNG)
    btnDownloadShap.addEventListener("click", () => {
        if (!shapChart) {
            alert("Không tìm thấy biểu đồ SHAP nào để tải!");
            return;
        }
        try {
            const imageURI = shapChart.toBase64Image();
            const link = document.createElement("a");
            link.download = `shap_chart_${patientIdInput.value.trim() || 'strain'}.png`;
            link.href = imageURI;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (err) {
            console.error("Lỗi khi tải biểu đồ SHAP:", err);
            alert("Không thể tải biểu đồ dưới dạng ảnh. Bạn hãy nhấn chuột phải vào biểu đồ và chọn 'Save image as...'.");
        }
    });

    // ----------------- LỊCH SỬ CHẨN ĐOÁN (SQLITE BACKEND) -----------------
    async function loadHistory() {
        try {
            const res = await fetch("/api/history");
            const data = await res.json();
            if (data.status === "success") {
                renderHistory(data.history);
            } else {
                historyTableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: #ef4444; padding: 2rem;">Không tải được lịch sử: ${data.message}</td></tr>`;
            }
        } catch (err) {
            console.error("Lỗi tải lịch sử:", err);
            historyTableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: #ef4444; padding: 2rem;">Lỗi kết nối đến máy chủ.</td></tr>`;
        }
    }

    function renderHistory(history) {
        historyTableBody.innerHTML = "";
        if (history.length === 0) {
            historyTableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; padding: 3rem; color: var(--text-muted);">
                <i class="fa-solid fa-folder-open" style="font-size: 2.5rem; display: block; margin-bottom: 1rem; opacity: 0.3;"></i>
                Chưa ghi nhận lịch sử phân tích nào trong cơ sở dữ liệu.
            </td></tr>`;
            return;
        }

        history.forEach(item => {
            const tr = document.createElement("tr");
            
            const probPct = (item.probability * 100).toFixed(1) + "%";
            const badgeClass = item.prediction === "Resistant" ? "label-resistant" : "label-susceptible";
            const badgeText = item.prediction === "Resistant" ? "Kháng Ciprofloxacin" : "Nhạy cảm";
            
            tr.innerHTML = `
                <td>${item.timestamp}</td>
                <td><strong style="color: #cbd5e1;">${item.strain_id || item.patient_id}</strong></td>
                <td><span class="badge ${badgeClass}">${badgeText}</span></td>
                <td><strong style="color: ${item.prediction === 'Resistant' ? '#f87171' : '#34d399'};">${probPct}</strong></td>
                <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${item.detected_genes}">
                    ${item.detected_genes}
                </td>
                <td>
                    <button class="btn btn-outline btn-sm btn-review" style="padding: 0.4rem 0.8rem; font-size: 0.8rem; height: auto;">
                        <i class="fa-solid fa-eye"></i> Xem lại
                    </button>
                    <button class="btn btn-outline-danger btn-sm btn-delete-item" style="padding: 0.4rem 0.8rem; font-size: 0.8rem; height: auto; margin-left: 0.25rem;">
                        <i class="fa-solid fa-trash"></i> Xóa
                    </button>
                </td>
            `;
            
            // Xem lại bản ghi cũ
            tr.querySelector(".btn-review").addEventListener("click", () => {
                jsonInput.value = JSON.stringify(item.features, null, 2);
                patientIdInput.value = item.strain_id || item.patient_id;
                switchTab("tab-single");
                btnPredictSingle.click();
            });
            
            // Xóa bản ghi
            tr.querySelector(".btn-delete-item").addEventListener("click", async () => {
                if (confirm(`Bạn có chắc chắn muốn xóa lịch sử của chủng vi khuẩn ${item.strain_id || item.patient_id}?`)) {
                    try  {
                        const res = await fetch("/api/history/delete", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ id: item.id })
                        });
                        const resData = await res.json();
                        if (resData.status === "success") {
                            loadHistory();
                        } else {
                            alert("Lỗi khi xóa: " + resData.message);
                        }
                    } catch (err) {
                        alert("Lỗi kết nối: " + err);
                    }
                }
            });

            historyTableBody.appendChild(tr);
        });
    }

    btnClearHistory.addEventListener("click", async () => {
        if (confirm("CẢNH BÁO NGUY HIỂM: Bạn có chắc chắn muốn xóa TOÀN BỘ lịch sử phân tích chủng? Thao tác này sẽ dọn sạch cơ sở dữ liệu SQLite và không thể khôi phục.")) {
            try {
                const res = await fetch("/api/history/clear", {
                    method: "POST"
                });
                const resData = await res.json();
                if (resData.status === "success") {
                    loadHistory();
                } else {
                    alert("Lỗi khi xóa toàn bộ lịch sử: " + resData.message);
                }
            } catch (err) {
                alert("Lỗi kết nối: " + err);
            }
        }
    });

    // Tải lịch sử chẩn đoán ban đầu
    loadHistory();

    // Sự kiện in báo cáo y khoa / Lưu PDF
    const btnPrintReport = document.getElementById("btn-print-report");
    btnPrintReport.addEventListener("click", () => {
        window.print();
    });

    // ----------------- DỊCH TỄ HỌC (EPIDEMIOLOGY STATS) -----------------
    let epiTrendChart = null;
    let epiGeneChart = null;

    async function loadEpidemiologyStats() {
        try {
            const res = await fetch("/api/epidemiology_stats");
            const data = await res.json();
            if (data.status === "success") {
                drawEpidemiologyCharts(data.timeline, data.genes);
            } else {
                console.error("Không tải được dữ liệu dịch tễ:", data.message);
            }
        } catch (err) {
            console.error("Lỗi kết nối API dịch tễ:", err);
        }
    }

    function drawEpidemiologyCharts(timeline, genes) {
        // 1. Vẽ biểu đồ xu hướng (Line Chart)
        const ctxTrend = document.getElementById("epiTrendChart").getContext("2d");
        if (epiTrendChart) {
            epiTrendChart.destroy();
        }
        
        const labelsTrend = timeline.map(t => t.date);
        const dataRates = timeline.map(t => t.rate);
        
        epiTrendChart = new Chart(ctxTrend, {
            type: 'line',
            data: {
                labels: labelsTrend,
                datasets: [{
                    label: 'Tỷ lệ kháng thuốc (%)',
                    data: dataRates,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3,
                    pointBackgroundColor: '#ef4444',
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Tỷ lệ kháng: ${context.raw}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8', callback: value => value + "%" }
                    }
                }
            }
        });

        // 2. Vẽ biểu đồ phân bố gen kháng thuốc (Bar Chart)
        const ctxGene = document.getElementById("epiGeneChart").getContext("2d");
        if (epiGeneChart) {
            epiGeneChart.destroy();
        }
        
        const topGenes = genes.slice(0, 10);
        const labelsGene = topGenes.map(g => g.gene);
        const dataCounts = topGenes.map(g => g.count);
        
        epiGeneChart = new Chart(ctxGene, {
            type: 'bar',
            data: {
                labels: labelsGene,
                datasets: [{
                    label: 'Số ca phát hiện',
                    data: dataCounts,
                    backgroundColor: 'rgba(99, 102, 241, 0.75)',
                    borderColor: '#6366f1',
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8', stepSize: 1 }
                    }
                }
            }
        });
    }
 
    loadGeneDictionary();
});

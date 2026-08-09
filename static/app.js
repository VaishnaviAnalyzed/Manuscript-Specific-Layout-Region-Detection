const uploadForm = document.getElementById("uploadForm");
const imageInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const selectedFile = document.getElementById("selectedFile");
const filePreview = document.getElementById("filePreview");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const removeFile = document.getElementById("removeFile");
const analyzeButton = document.getElementById("analyzeButton");
const statusMessage = document.getElementById("statusMessage");
const resultsSection = document.getElementById("resultsSection");
const resultImage = document.getElementById("resultImage");
const regionCount = document.getElementById("regionCount");
const regionList = document.getElementById("regionList");
const metadataLink = document.getElementById("metadataLink");

const classColors = {
    header: "#3182f6",
    footer: "#ea580c",
    main_text: "#16a34a",
    side_text: "#a855f7",
    filler: "#d8a807",
};

let currentFile = null;
let previewUrl = null;

function readableSize(bytes) {
    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showSelectedFile(file) {
    if (!file.type.startsWith("image/")) {
        statusMessage.textContent = "Please select a supported image file.";
        return;
    }
    if (file.size > 20 * 1024 * 1024) {
        statusMessage.textContent = "The selected image is larger than 20 MB.";
        return;
    }

    currentFile = file;
    statusMessage.textContent = "";
    fileName.textContent = file.name;
    fileSize.textContent = readableSize(file.size);
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }
    previewUrl = URL.createObjectURL(file);
    filePreview.src = previewUrl;
    selectedFile.classList.remove("hidden");
    analyzeButton.disabled = false;
}

function clearSelectedFile() {
    currentFile = null;
    imageInput.value = "";
    selectedFile.classList.add("hidden");
    analyzeButton.disabled = true;
    statusMessage.textContent = "";
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }
}

function renderRegions(regions) {
    regionList.innerHTML = "";
    regions.forEach((region) => {
        const item = document.createElement("div");
        item.className = "region-item";
        const box = region.bbox;
        const label = region.label.replace("_", " ");
        item.innerHTML = `
            <span class="region-dot" style="background:${classColors[region.label]}"></span>
            <div>
                <strong>${label}</strong>
                <small>x ${box.x_min}–${box.x_max}, y ${box.y_min}–${box.y_max}</small>
            </div>
            <span class="confidence">${Math.round(region.confidence * 100)}%</span>
        `;
        regionList.appendChild(item);
    });
}

imageInput.addEventListener("change", () => {
    if (imageInput.files.length > 0) {
        showSelectedFile(imageInput.files[0]);
    }
});

removeFile.addEventListener("click", clearSelectedFile);

["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("dragging");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove("dragging");
    });
});

dropZone.addEventListener("drop", (event) => {
    if (event.dataTransfer.files.length > 0) {
        showSelectedFile(event.dataTransfer.files[0]);
    }
});

uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!currentFile) {
        return;
    }

    const formData = new FormData();
    formData.append("file", currentFile);
    analyzeButton.disabled = true;
    analyzeButton.classList.add("loading");
    analyzeButton.querySelector("span").textContent = "Analyzing page";
    statusMessage.textContent = "";

    try {
        const response = await fetch("/api/detect", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Detection failed.");
        }

        resultImage.src = `${data.annotated_image_url}?v=${Date.now()}`;
        metadataLink.href = data.metadata_url;
        regionCount.textContent = data.region_count;
        renderRegions(data.regions);
        resultsSection.classList.remove("hidden");
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
        statusMessage.textContent = error.message || "Could not connect to the server.";
    } finally {
        analyzeButton.disabled = false;
        analyzeButton.classList.remove("loading");
        analyzeButton.querySelector("span").textContent = "Detect regions";
    }
});


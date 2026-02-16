const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const log = document.getElementById("statusLog"); // may be null
const predictBtn = document.getElementById("predictBtn");

let recordingTimer = null;
let recordingSeconds = 0;
let mediaRecorder;
let recordedChunks = [];
let recordedBlob = null;

/* ================= SAFE LOGGER ================= */
function addLog(msg, error = false) {
  if (!log) return; // 🔥 PREVENT CRASH
  const li = document.createElement("li");
  li.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  if (error) li.style.color = "red";
  log.prepend(li);
}

/* ================= CAMERA ================= */
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: true
    });

    video.srcObject = stream;
    video.controls = false;
    video.play();

    document.getElementById("videoActions").classList.add("hidden");
    document.getElementById("recordControls").classList.remove("hidden");

    overlay.classList.add("hidden");
    addLog("Camera started");
  } catch (err) {
    addLog("Camera access failed", true);
  }
}

/* ================= RECORDING ================= */
function startRecording() {
  try {
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(video.srcObject);

    mediaRecorder.ondataavailable = e => recordedChunks.push(e.data);

    mediaRecorder.onstop = () => {
      recordedBlob = new Blob(recordedChunks, { type: "video/webm" });

      // 🔥 IMPORTANT: SHOW RECORDED VIDEO
      video.srcObject = null;
      video.src = URL.createObjectURL(recordedBlob);
      video.controls = true;
      video.play();

      stopRecordingTimer();
      overlay.classList.add("hidden");
      predictBtn.disabled = false;

      addLog("Recording completed");
    };

    mediaRecorder.start();
    startRecordingTimer();

    overlay.innerText = "Recording...";
    overlay.classList.remove("hidden");
    addLog("Recording started");
  } catch (err) {
    addLog("Recording error", true);
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    overlay.innerText = "Stopping...";
    overlay.classList.remove("hidden");
    addLog("Stopping recording");
  }
}

/* ================= UPLOAD ================= */
document.getElementById("upload").addEventListener("change", e => {
  recordedBlob = e.target.files[0];

  video.srcObject = null;
  video.src = URL.createObjectURL(recordedBlob);
  video.controls = true;
  video.play();

  document.getElementById("videoActions").classList.add("hidden");
  document.getElementById("recordControls").classList.remove("hidden");

  overlay.classList.add("hidden");
  predictBtn.disabled = false;

  addLog("Video uploaded");
});

/* ================= PREDICT ================= */
async function predict() {
  if (!recordedBlob) {
    addLog("No video to predict", true);
    return;
  }

  try {
    overlay.innerText = "Analyzing...";
    overlay.classList.remove("hidden");
    predictBtn.disabled = true;

    addLog("Prediction started");

    const data = await sendToBackend(recordedBlob);
    showResult(data);

    overlay.classList.add("hidden");
    predictBtn.disabled = false;

    addLog("Prediction successful");
  } catch (err) {
    overlay.innerText = "Prediction Failed";
    addLog("Prediction error", true);
    predictBtn.disabled = false;
  }
}

/* ================= RESULT ================= */
function showResult(data) {
  document.getElementById("result").classList.remove("hidden");

  // ===== SCORE LOGIC (UNCHANGED) =====
  const truthProb = data.result.truth_probability;
  const lieProb = data.result.lie_probability;

  const meanLieProb = 0.70;
  let score = 50 + ((lieProb - meanLieProb) / (1 - meanLieProb)) * 50;
  score = Math.max(0, Math.min(100, Math.round(score)));

  function getLabel(score) {
    if (score <= 10) return "Strongly Truth";
    if (score <= 50) return "Truth";
    if (score <= 60) return "Goodly Truth";
    if (score <= 75) return "Uncertain";
    if (score <= 85) return "Cannot Say (50–50)";
    if (score <= 90) return "Lie";
    return "Strongly Lie";
  }

  document.getElementById("finalScore").innerText = `Score: ${score}/100`;
  document.getElementById("finalLabel").innerText = getLabel(score);

  // ===== INFO PANEL CONTENT =====
  document.getElementById("infoContent").innerText =
    JSON.stringify(data, null, 2);
}

/* ================= RECORDING TIMER ================= */
function startRecordingTimer() {
  recordingSeconds = 0;
  const timerEl = document.getElementById("recordingTimer");
  const dotEl = document.getElementById("recordingDot");

  dotEl.classList.remove("hidden");
  timerEl.classList.remove("hidden");

  recordingTimer = setInterval(() => {
    recordingSeconds++;
    const m = String(Math.floor(recordingSeconds / 60)).padStart(2, "0");
    const s = String(recordingSeconds % 60).padStart(2, "0");
    timerEl.innerText = `${m}:${s}`;
  }, 1000);
}

function stopRecordingTimer() {
  clearInterval(recordingTimer);
  document.getElementById("recordingDot").classList.add("hidden");
  document.getElementById("recordingTimer").classList.add("hidden");
}
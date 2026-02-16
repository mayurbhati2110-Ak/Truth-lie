const NGROK_API = "https://intertentacular-fallaciously-annett.ngrok-free.dev/predict";

async function sendToBackend(videoBlob) {
  const formData = new FormData();

  // 🔥 MUST be "file" to match FastAPI
  formData.append("file", videoBlob, "video.webm");

  const response = await fetch(NGROK_API, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(err);
  }

  return await response.json();
}
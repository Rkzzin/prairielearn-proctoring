async function loadConfig() {
  const response = await fetch(chrome.runtime.getURL("config.json"), { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Configuração de allowlist indisponível.");
  }
  return response.json();
}

function normalizeHost(value) {
  const raw = value.trim().toLowerCase();
  if (!raw) return "";
  try {
    const url = raw.includes("://") ? new URL(raw) : new URL(`https://${raw}`);
    return url.hostname.replace(/^\*\./, "");
  } catch {
    return raw.replace(/^\*\./, "").split("/")[0];
  }
}

function isAllowed(host, sites) {
  return sites.some((site) => host === site.host || host.endsWith(`.${site.host}`));
}

function destinationFor(value, sites) {
  const raw = value.trim();
  const host = normalizeHost(raw);
  if (!host || !isAllowed(host, sites)) return null;
  if (raw.includes("://")) return raw;
  return `https://${host}`;
}

function renderSites(config) {
  const container = document.querySelector("#sites");
  container.textContent = "";
  for (const site of config.sites || []) {
    const anchor = document.createElement("a");
    anchor.href = site.url;
    anchor.className = "site-card";
    anchor.innerHTML = `<strong>${site.label}</strong><span>${site.host}</span>`;
    container.appendChild(anchor);
  }
  if (!container.children.length) {
    container.textContent = "Nenhum site permitido foi configurado.";
  }
}

async function main() {
  const config = await loadConfig();
  const sites = config.sites || [];
  renderSites(config);

  const form = document.querySelector("#go-form");
  const input = document.querySelector("#target");
  const message = document.querySelector("#message");

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const destination = destinationFor(input.value, sites);
    if (!destination) {
      message.textContent = "Esse destino não está na allowlist da prova.";
      return;
    }
    window.location.assign(destination);
  });
}

main().catch((error) => {
  document.querySelector("#message").textContent = error.message;
});

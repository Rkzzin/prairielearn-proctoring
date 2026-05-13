const CONFIG_PATH = "config.json";
const BLOCK_RULE_ID = 1;
const ALLOW_RULE_START_ID = 100;
let activeConfig = null;

async function loadConfig() {
  const response = await fetch(chrome.runtime.getURL(CONFIG_PATH), { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Falha ao carregar ${CONFIG_PATH}: ${response.status}`);
  }
  return response.json();
}

function ruleForAllowedSite(site, index, resourceTypes) {
  return {
    id: ALLOW_RULE_START_ID + index,
    priority: 100,
    action: { type: "allow" },
    condition: {
      requestDomains: [site.host],
      resourceTypes,
    },
  };
}

function blockRule(resourceTypes) {
  return {
    id: BLOCK_RULE_ID,
    priority: 1,
    action: {
      type: "redirect",
      redirect: { extensionPath: "/blocked.html" },
    },
    condition: {
      regexFilter: "^https?://",
      resourceTypes,
    },
  };
}

function normalizeHost(value) {
  try {
    return new URL(value).hostname.toLowerCase().replace(/\.$/, "");
  } catch {
    return "";
  }
}

function isExtensionUrl(url) {
  return url.startsWith(chrome.runtime.getURL(""));
}

function isAllowedUrl(url, config) {
  if (!url || isExtensionUrl(url) || url === "about:blank") {
    return true;
  }

  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return false;
  }

  if (!["http:", "https:"].includes(parsed.protocol)) {
    return false;
  }

  const host = normalizeHost(url);
  return (config.sites || []).some((site) => {
    return host === site.host || host.endsWith(`.${site.host}`);
  });
}

function blockedUrl(originalUrl) {
  return chrome.runtime.getURL(`blocked.html?url=${encodeURIComponent(originalUrl || "")}`);
}

async function enforceTab(tabId, url) {
  const config = activeConfig || await loadConfig();
  activeConfig = config;
  if (isAllowedUrl(url, config)) {
    return;
  }
  await chrome.tabs.update(tabId, { url: blockedUrl(url) });
}

async function enforceOpenTabs() {
  const tabs = await chrome.tabs.query({});
  await Promise.all(
    tabs
      .filter((tab) => typeof tab.id === "number" && typeof tab.url === "string")
      .map((tab) => enforceTab(tab.id, tab.url).catch((error) => console.error(error)))
  );
}

async function applyRules() {
  const config = await loadConfig();
  activeConfig = config;
  await chrome.storage.local.set({ allowlistConfig: config });

  const current = await chrome.declarativeNetRequest.getSessionRules();
  const removeRuleIds = current.map((rule) => rule.id);
  const resourceTypes = config.strictSubresources
    ? ["main_frame", "sub_frame", "xmlhttprequest", "script", "image", "stylesheet", "font", "media", "object", "other"]
    : ["main_frame"];

  const addRules = [
    blockRule(resourceTypes),
    ...(config.sites || []).map((site, index) => ruleForAllowedSite(site, index, resourceTypes)),
  ];

  await chrome.declarativeNetRequest.updateSessionRules({
    removeRuleIds,
    addRules,
  });
  await enforceOpenTabs();
}

chrome.runtime.onInstalled.addListener(() => {
  applyRules().catch((error) => console.error(error));
});

chrome.runtime.onStartup.addListener(() => {
  applyRules().catch((error) => console.error(error));
});

chrome.tabs.onCreated.addListener(() => {
  applyRules().catch((error) => console.error(error));
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  const url = changeInfo.url || tab.url;
  if (!url) {
    return;
  }
  enforceTab(tabId, url).catch((error) => console.error(error));
});

chrome.webNavigation.onBeforeNavigate.addListener((details) => {
  if (details.frameId !== 0) {
    return;
  }
  enforceTab(details.tabId, details.url).catch((error) => console.error(error));
});

applyRules().catch((error) => console.error(error));

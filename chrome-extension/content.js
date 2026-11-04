function extractVisibleComments() {
  return [...document.querySelectorAll("ytd-comment-thread-renderer #content-text")]
    .map((node) => node.innerText.trim()).filter(Boolean).slice(0, 100);
}
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "extractComments") sendResponse({ comments: extractVisibleComments() });
  return true;
});

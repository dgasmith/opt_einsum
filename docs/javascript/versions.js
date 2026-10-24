// Keep Material's version navigation, with a separate entry for the stable alias.
function labelVersions() {
  const selector = document.querySelector(".md-version");
  if (!selector) return false;
  if (selector.dataset.labelled) return true;
  selector.dataset.labelled = "true";

  const list = selector.querySelector(".md-version__list");
  const stable = Array.from(list.children).find(item =>
    item.querySelector(".md-version__alias")?.textContent === "stable"
  );
  if (stable) {
    const latest = stable.cloneNode(true);
    latest.querySelector("a").textContent = "Latest (stable)";
    list.prepend(latest);
  }
  const current = selector.querySelector(".md-version__current");
  if (current.querySelector(".md-version__alias")?.textContent === "stable") {
    current.textContent = "Latest (stable)";
  }
  selector.querySelectorAll(".md-version__alias").forEach(alias => alias.remove());
  return true;
}

// Material creates its selector after fetching Mike's versions.json.
document$.subscribe(() => {
  if (labelVersions()) return;
  const header = document.querySelector(".md-header__topic");
  if (!header) return;
  const observer = new MutationObserver(() => {
    if (labelVersions()) observer.disconnect();
  });
  observer.observe(header, { childList: true, subtree: true });
});

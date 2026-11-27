// Tooltips for chart marks, on hover and on keyboard focus.
(function () {
  var tip = document.getElementById("tip");
  function show(e) {
    var el = e.target.closest("[data-tip]");
    if (!el) return;
    tip.textContent = el.dataset.tip;
    tip.hidden = false;
    var r = el.getBoundingClientRect();
    var left = Math.min(Math.max(8, r.left + r.width / 2 - tip.offsetWidth / 2), innerWidth - tip.offsetWidth - 8);
    tip.style.left = left + scrollX + "px";
    tip.style.top = r.top + scrollY - tip.offsetHeight - 8 + "px";
  }
  function hide() { tip.hidden = true; }
  document.addEventListener("pointerover", show);
  document.addEventListener("pointerout", hide);
  document.addEventListener("focusin", show);
  document.addEventListener("focusout", hide);
})();

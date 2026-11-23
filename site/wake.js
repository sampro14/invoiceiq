// Polls the sleeping app and moves the visitor on as soon as it answers.
(function () {
  var app = document.body.dataset.app;
  var status = document.getElementById("status");
  var bar = document.getElementById("bar");
  var started = Date.now();
  function attempt() {
    var seconds = (Date.now() - started) / 1000;
    bar.style.width = Math.min(96, 100 * (1 - Math.exp(-seconds / 28))) + "%";
    if (seconds > 150) {
      status.textContent = "The agent is not answering. Try opening the app directly, or come back in a few minutes.";
      return;
    }
    fetch(app + "/healthz", { cache: "no-store" }).then(function (r) {
      if (!r.ok) throw new Error("not ready");
      bar.style.width = "100%";
      status.textContent = "Ready. Opening the app.";
      location.replace(app);
    }).catch(function () { setTimeout(attempt, 2000); });
  }
  attempt();
})();

(function () {
  "use strict";

  const state = {};

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((candidate) => {
        const active = candidate === tab;
        candidate.classList.toggle("active", active);
        candidate.setAttribute("aria-selected", String(active));
      });
      document.querySelectorAll(".panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === `panel-${tab.dataset.tab}`);
      });
    });
  });

  document.querySelectorAll("[data-dropzone]").forEach((zone) => {
    const input = zone.querySelector('input[type="file"]');
    const label = zone.querySelector(".file-name");
    const update = () => {
      label.textContent = input.files?.[0]?.name || "Ningún archivo seleccionado";
    };
    input.addEventListener("change", update);
    ["dragenter", "dragover"].forEach((eventName) => {
      zone.addEventListener(eventName, (event) => {
        event.preventDefault();
        zone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      zone.addEventListener(eventName, (event) => {
        event.preventDefault();
        zone.classList.remove("dragover");
      });
    });
    zone.addEventListener("drop", (event) => {
      if (event.dataTransfer?.files?.length) {
        input.files = event.dataTransfer.files;
        update();
      }
    });
  });

  document.querySelectorAll("form[data-kind]").forEach((form) => {
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      startJob(form);
    });
  });

  document.querySelectorAll("[data-action='delete']").forEach((button) => {
    button.addEventListener("click", () => discard(button.closest(".job")));
  });

  async function startJob(form) {
    const kind = form.dataset.kind;
    const endpoint = form.dataset.endpoint;
    const box = document.getElementById(`${kind}-job`);
    const submit = form.querySelector("button.primary");

    resetJob(box);
    box.hidden = false;
    box.querySelector(".progress").classList.add("indeterminate");
    setStatus(box, "Subiendo archivo...");
    submit.disabled = true;

    try {
      const response = await fetch(endpoint, { method: "POST", body: new FormData(form) });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || `Error HTTP ${response.status}`);
      }
      setServerStatus(true);
      box.dataset.jobId = data.job_id;
      state[kind] = { jobId: data.job_id, timer: null };
      poll(kind, form);
    } catch (error) {
      setServerStatus(false);
      showError(box, error.message || "No se pudo iniciar el trabajo.");
      box.querySelector(".progress").classList.remove("indeterminate");
      submit.disabled = false;
    }
  }

  async function poll(kind, form) {
    const job = state[kind];
    const box = document.getElementById(`${kind}-job`);
    if (!job) {
      return;
    }

    try {
      const response = await fetch(`/api/jobs/${job.jobId}`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || `Error HTTP ${response.status}`);
      }

      setServerStatus(true);
      renderProgress(box, data);

      if (data.state === "done") {
        renderResult(box, data);
        job.timer = null;
        form.querySelector("button.primary").disabled = false;
        return;
      }
      if (data.state === "error") {
        showError(box, data.error || "El trabajo falló.");
        job.timer = null;
        form.querySelector("button.primary").disabled = false;
        return;
      }

      job.timer = setTimeout(() => poll(kind, form), 1000);
    } catch (error) {
      setServerStatus(false);
      showError(box, error.message || "Se perdió la conexión con el servidor.");
      job.timer = null;
      form.querySelector("button.primary").disabled = false;
    }
  }

  function renderProgress(box, job) {
    const progress = box.querySelector(".progress");
    const bar = box.querySelector(".progress-bar");
    const status = box.querySelector(".status");

    if (typeof job.progress === "number") {
      progress.classList.remove("indeterminate");
      bar.style.width = `${Math.round(job.progress * 100)}%`;
    } else {
      progress.classList.add("indeterminate");
      bar.style.width = "";
    }
    status.textContent = job.message || "Procesando...";
    status.hidden = false;
  }

  function renderResult(box, job) {
    const result = job.result || {};
    const container = box.querySelector(".result");
    container.hidden = false;
    container.replaceChildren();

    container.append(el("h3", { text: "Resultado" }));

    if (job.kind === "frames") {
      renderFrames(container, job, result);
    } else {
      renderTranscription(container, job, result);
    }
  }

  function renderFrames(container, job, result) {
    const summary = el("div", { className: "summary" }, [
      badge(`${result.frame_count || 0} capturas`, true),
    ]);
    container.append(summary);

    const actions = el("div", { className: "actions" });
    if (result.artifacts?.zip) {
      actions.append(downloadLink(job.id, "zip", "Descargar ZIP"));
    }
    if (result.artifacts?.csv) {
      actions.append(downloadLink(job.id, "csv", "Descargar CSV", true));
    }
    container.append(actions);

    if (result.frames?.length) {
      const gallery = el("div", { className: "gallery" });
      result.frames.forEach((name) => {
        const image = el("img", {
          attrs: {
            src: `/api/jobs/${job.id}/frames/${encodeURIComponent(name)}`,
            alt: name,
            loading: "lazy",
          },
        });
        gallery.append(
          el("figure", { className: "figure" }, [image, el("figcaption", { text: name })]),
        );
      });
      container.append(gallery);
    }

    if (result.index?.length) {
      container.append(
        el("h3", { text: "Índice de capturas" }),
        buildTable(
          ["Archivo", "Tiempo (s)", "HH:MM:SS"],
          result.index.map((row) => [
            row.archivo,
            row.tiempo_aproximado_segundos,
            row.tiempo_hhmmss,
          ]),
        ),
      );
    }
  }

  function renderTranscription(container, job, result) {
    const summary = el("div", { className: "summary" });
    summary.append(badge(`${result.segment_count || 0} segmentos`, true));
    if (result.device) {
      summary.append(badge(`${result.device} / ${result.compute_type || "auto"}`));
    }
    if (result.language) {
      const probability =
        typeof result.language_probability === "number"
          ? ` ${(result.language_probability * 100).toFixed(1)}%`
          : "";
      summary.append(badge(`Idioma: ${result.language}${probability}`));
    }
    if (typeof result.duration === "number") {
      summary.append(badge(`Duración: ${formatDuration(result.duration)}`));
    }
    container.append(summary);

    const actions = el("div", { className: "actions" });
    if (result.artifacts?.txt) {
      actions.append(downloadLink(job.id, "txt", "Descargar TXT"));
    }
    if (result.artifacts?.srt) {
      actions.append(downloadLink(job.id, "srt", "Descargar SRT", true));
    }
    container.append(actions);

    container.append(
      el("h3", { text: "Texto completo" }),
      el("pre", { className: "transcript", text: result.text || "Sin contenido." }),
    );

    if (result.segments?.length) {
      container.append(
        el("h3", { text: "Segmentos" }),
        buildTable(
          ["Tiempo", "Texto"],
          result.segments.map((segment) => [segment.time, segment.text]),
        ),
      );
    }
  }

  function buildTable(headers, rows) {
    const head = el("thead");
    const headRow = el("tr");
    headers.forEach((header) => headRow.append(el("th", { text: header })));
    head.append(headRow);

    const body = el("tbody");
    rows.forEach((row) => {
      const tr = el("tr");
      row.forEach((value) => tr.append(el("td", { text: value ?? "" })));
      body.append(tr);
    });

    return el("div", { className: "table-wrap" }, [el("table", {}, [head, body])]);
  }

  function badge(text, ok = false) {
    return el("span", { className: ok ? "badge ok" : "badge", text });
  }

  function downloadLink(jobId, artifact, label, secondary = false) {
    return el("a", {
      className: secondary ? "secondary" : "",
      text: label,
      attrs: {
        href: `/api/jobs/${jobId}/download/${artifact}`,
        download: "",
      },
    });
  }

  function el(tag, options = {}, children = []) {
    const node = document.createElement(tag);
    if (options.className) {
      node.className = options.className;
    }
    if (options.text !== undefined) {
      node.textContent = options.text;
    }
    if (options.attrs) {
      Object.entries(options.attrs).forEach(([key, value]) => node.setAttribute(key, value));
    }
    children.filter(Boolean).forEach((child) => node.append(child));
    return node;
  }

  function formatDuration(seconds) {
    const total = Math.max(0, Math.round(seconds));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const rest = total % 60;
    const pad = (value) => String(value).padStart(2, "0");
    return hours ? `${hours}:${pad(minutes)}:${pad(rest)}` : `${minutes}:${pad(rest)}`;
  }

  async function discard(box) {
    if (!box) {
      return;
    }
    const kind = box.id.replace("-job", "");
    const jobId = box.dataset.jobId;
    if (state[kind]?.timer) {
      clearTimeout(state[kind].timer);
    }
    delete state[kind];
    if (jobId) {
      try {
        await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
      } catch (error) {
        setServerStatus(false);
      }
    }
    box.dataset.jobId = "";
    resetJob(box);
    box.hidden = true;
  }

  function resetJob(box) {
    const progress = box.querySelector(".progress");
    progress.classList.remove("indeterminate");
    box.querySelector(".progress-bar").style.width = "";
    box.querySelector(".status").textContent = "En cola";
    box.querySelector(".status").hidden = false;
    const error = box.querySelector(".error");
    error.hidden = true;
    error.textContent = "";
    const result = box.querySelector(".result");
    result.hidden = true;
    result.replaceChildren();
  }

  function showError(box, message) {
    const error = box.querySelector(".error");
    error.hidden = false;
    error.textContent = message;
    const status = box.querySelector(".status");
    status.textContent = "Sin progreso";
    status.hidden = true;
  }

  function setStatus(box, message) {
    const status = box.querySelector(".status");
    status.hidden = false;
    status.textContent = message;
  }

  function setServerStatus(connected) {
    const status = document.getElementById("server-status");
    if (status) {
      status.textContent = connected ? "conectado" : "sin conexión";
    }
  }
})();

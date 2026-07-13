/* Lucide icon paths, ISC License: https://lucide.dev/license */
(function () {
  const paths = {
    "archive": '<rect width="20" height="5" x="2" y="3" rx="1"/><path d="M4 8v11a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8M10 12h4"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "check-circle": '<path d="M22 11.1V12a10 10 0 1 1-5.9-9.1"/><path d="m9 11 3 3L22 4"/>',
    "chevron-down": '<path d="m6 9 6 6 6-6"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5M12 15V3"/>',
    "file-text": '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><path d="M8 13h8M8 17h8M8 9h2"/>',
    "flask-conical": '<path d="M10 2v7.31M14 9.3V1.99M8.5 2h7"/><path d="M14 9.3 19.7 19a2 2 0 0 1-1.7 3H6a2 2 0 0 1-1.7-3L10 9.3"/><path d="M6.5 16h11"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    "key-round": '<path d="M2.586 17.414A2 2 0 0 0 4 18h2v2h2v2h4l3.586-3.586A2 2 0 0 0 16.172 17H18a4 4 0 1 0-3.874-5H12.83a2 2 0 0 0-1.414.586z"/><circle cx="20" cy="8" r=".5" fill="currentColor"/>',
    "lock": '<rect width="18" height="11" x="3" y="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "refresh-cw": '<path d="M21 12a9 9 0 0 0-15.2-6.5L3 8"/><path d="M3 3v5h5M3 12a9 9 0 0 0 15.2 6.5L21 16"/><path d="M16 16h5v5"/>',
    "save": '<path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M17 21v-8H7v8M7 3v5h8"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "terminal": '<polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/>',
    "trash-2": '<path d="M3 6h18M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M10 11v6M14 11v6"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5M12 3v12"/>',
    "x": '<path d="M18 6 6 18M6 6l12 12"/>'
  };

  function icon(name, className) {
    const body = paths[name] || paths.info;
    return `<svg class="lucide ${className || ""}" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;
  }

  function hydrate(root) {
    (root || document).querySelectorAll("[data-icon]").forEach((element) => {
      if (element.querySelector("svg.lucide")) return;
      element.insertAdjacentHTML("afterbegin", icon(element.dataset.icon));
    });
  }

  window.GemsIcons = { icon, hydrate };
})();

const container = document.getElementById("transcript-container");

fetch("../_assets/tale_two_cities_ch01.txt")
  .then((r) => r.text())
  .then((text) => {
    text.split(/\n\n+/).forEach((paragraph) => {
      const p = document.createElement("p");
      p.className = "chunk";
      p.textContent = paragraph.replace(/\n/g, " ");
      container.appendChild(p);
    });
  });

export default {
  root: "src",
  title: "ACBUDDY Explorer",
  base: "/explore/",
  preserveExtension: true,
  preserveIndex: true,
  sidebar: false,
  toc: false,
  pager: false,
  header: false,
  footer: false,
  head: `<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">`,
  style: "custom.css",
  pages: [
    {name: "Archive", path: "/"},
    {name: "Research Deep Dive", path: "/research/"},
    {name: "Knowledge Graph", path: "/graph"},
    {name: "Performance", path: "/performance"}
  ]
};

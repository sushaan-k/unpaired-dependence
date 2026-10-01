const dialog = document.querySelector('.figure-dialog');
const image = dialog.querySelector('img');

document.querySelectorAll('[data-figure]').forEach(link => {
  link.addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    image.src = link.href;
    image.alt = link.querySelector('img').alt;
    dialog.showModal();
  });
});

dialog.querySelector('button').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  if (event.target === dialog) dialog.close();
});

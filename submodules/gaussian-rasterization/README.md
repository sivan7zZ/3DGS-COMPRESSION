# Differential Gaussian Rasterization

This repository uses the standard Gaussian backend with SPS per-Gaussian score
gradients. It builds one extension, `diff_gaussian_rasterization._C`, from
`normal/` and `conv.cu`. The scores input and its custom backward gradient are
required by PWSR spatial and frequency pruning; do not replace this backend with
the upstream 3DGS or LightGaussian rasterizer.

Build from the main project directory in the selected PyTorch/CUDA environment:

```bash
python -m pip install ./submodules/gaussian-rasterization --no-build-isolation --no-deps
```

For an in-place build used by local development:

```bash
cd submodules/gaussian-rasterization
python setup.py build_ext --inplace
```

The CMake target builds only the standalone standard CUDA rasterizer. Use the
Python build above for the PyTorch extension and the SPS score-gradient interface.

Used as the rasterization engine for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". If you can make use of it in your own research, please be so kind to cite us.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>

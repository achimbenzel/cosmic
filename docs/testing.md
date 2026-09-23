# Testing

## Automated

```sh
cmake -S . -B build -DCOSMIC_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

`core` runs the unit tests, `plugin` runs the mock host against the built
`.aex` (Windows builds, or the mingw build through wine).

### Sanitizers

```sh
g++ -std=c++17 -O1 -g -fsanitize=address,undefined -I src -I tests \
    -o /tmp/ct tests/CoreTests.cpp tests/TestSupport.cpp src/core/*.cpp -pthread && /tmp/ct
g++ -std=c++17 -O1 -g -fsanitize=thread -I src -I tests \
    -o /tmp/ct tests/CoreTests.cpp tests/TestSupport.cpp src/core/*.cpp -pthread && /tmp/ct
```

### Looking at the output

`build/cosmic_preview <dir>` writes contact sheets of every palette, gradient
type, depth shape, turbulence setting and repeat mode, text renders (defaults,
focus, hot glow with and without highlight protection, inverted matte) and
timings at 1080p and 4K. The mock host writes `mockhost_*.png` for each case.

## In After Effects

The automated tests cover the pixels and the call sequence; these are the
things only the real host can tell you.

1. **It loads.** Copy `CosmicGradient.aex` to one plug-in folder, restart, and
   apply **Effect ▸ AB Tools ▸ Cosmic Gradient** to a text layer. No warning
   triangle in Effect Controls (that would mean After Effects does not consider
   the effect thread safe).
2. **The right build.** The last group is named `v1.0.0 (<commit>)`. After
   Effects scans both its own `Plug-ins` folder and
   `Common\Plug-ins\7.0\MediaCore`; a stale copy in the other one wins.
3. **Palettes.** Pick each palette: the five colours update. Edit a colour: the
   popup reads *Custom*. Undo restores both.
4. **Content bounds.** On a line of text the palette spans the text. *Fit:
   Layer* spreads it over the layer instead.
5. **Loop.** Keyframe Angle 0° → 360° over 5 s with turbulence on; render and
   check the first and last frames match and the loop has no jump.
6. **Focus.** Defocus 20, move the Focus Point across the text: sharp near it,
   soft away from it.
7. **Glow and protection.** Solar Flare, Glow Intensity 250 %: with protection
   the hot core stays cream-coloured; at 0 % it clips to white.
8. **Bit depths.** Switch the project between 8, 16 and 32 bpc; the look
   matches, and 32 bpc keeps values above 1.0 when protection is 0.
9. **Resolutions.** Full / Half / Quarter at a fixed viewer zoom: the gradient,
   glow and grain keep their size.
10. **Bounds.** With Expand Bounds on, the glow spills past the layer's edges;
    off, it stops at them. Full Frame never expands.
11. **Grain.** With Animate Grain off, scrubbing a static frame is cached (no
    re-render); on, the grain changes every frame.
12. **Multi-frame rendering.** Render a range from the render queue; frames
    render concurrently without artefacts.
13. **Stability.** Apply and remove the effect 20 times, undo/redo, duplicate
    the layer, save, reopen. Memory returns to where it was.

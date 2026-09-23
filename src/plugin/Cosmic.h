#pragma once

// Identity of the plug-in. These values are mirrored by tools/generate_pipl.py,
// which builds the PiPL resource; keep both in sync.
#define COSMIC_NAME "Cosmic Gradient"
#define COSMIC_MATCH_NAME "ABBZ CosmicGradient"
#define COSMIC_CATEGORY "AB Tools"
#define COSMIC_DESCRIPTION "Cinematic gradients for shapes and text, with glow and optical diffusion."

#define COSMIC_VERSION_MAJOR 1
#define COSMIC_VERSION_MINOR 0
#define COSMIC_VERSION_BUG 0
#define COSMIC_VERSION_BUILD 1

// Set by CMake from the git revision. Shown in the effect's parameters so the
// build After Effects actually loaded can be read off the UI.
#ifndef COSMIC_BUILD_ID
#define COSMIC_BUILD_ID "local"
#endif

#define COSMIC_STRINGIFY_(x) #x
#define COSMIC_STRINGIFY(x) COSMIC_STRINGIFY_(x)
#define COSMIC_VERSION_STRING                                                          \
    COSMIC_STRINGIFY(COSMIC_VERSION_MAJOR) "." COSMIC_STRINGIFY(COSMIC_VERSION_MINOR) "." \
        COSMIC_STRINGIFY(COSMIC_VERSION_BUG)

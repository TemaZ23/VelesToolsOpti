#!/usr/bin/env node
// Builds the Veles Tools extension from extension-build/:
//   app/bundle.js (editable 2.14.x app bundle) + src/*.mjs (new modules)
//   → dist/extension (unpacked) and release-artifacts/veles-tools-<version>.zip
//
// Usage: node extension-build/build.mjs [--test]
//   --test  also writes dist/test/app.js — the same bundle with internals exposed on
//           window.__VELES_TEST__ (used by the simulated end-to-end test, never shipped).

import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const root = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(root, '..');
const distDir = join(root, 'dist');
const extensionOut = join(distDir, 'extension');
const manifest = JSON.parse(readFileSync(join(root, 'extension', 'manifest.json'), 'utf8'));
const version = manifest.version;
const withTest = process.argv.includes('--test');

rmSync(distDir, { recursive: true, force: true });
mkdirSync(distDir, { recursive: true });
cpSync(join(root, 'extension'), extensionOut, { recursive: true });

const bundleName = `index-${version.replace(/\./g, '_')}.js`;
const result = await build({
  entryPoints: [join(root, 'app', 'bundle.js')],
  bundle: true,
  minify: true,
  format: 'esm',
  legalComments: 'none',
  logLevel: 'warning',
  outfile: join(extensionOut, 'ui', 'dist', 'assets', bundleName),
  metafile: true,
});
if (result.errors.length > 0) {
  process.exit(1);
}

const indexPath = join(extensionOut, 'ui', 'dist', 'index.html');
const indexHtml = readFileSync(indexPath, 'utf8').replace(/\.\/assets\/index-[^"]+\.js/, `./assets/${bundleName}`);
writeFileSync(indexPath, indexHtml);

if (withTest) {
  // Same app, but instead of rendering it exposes the optimizer internals.
  const source = readFileSync(join(root, 'app', 'bundle.js'), 'utf8');
  const marker = 'if (typeof window < "u" && window.self !== window.top)';
  if (!source.includes(marker)) {
    throw new Error('Render marker not found in app/bundle.js');
  }
  const testSource = source.replace(marker, `window.__VELES_TEST__ = __velesTestExports();\n${marker}`);
  const testEntry = join(root, 'app', '.bundle.test.js');
  writeFileSync(testEntry, `${testSource}\n${readFileSync(join(root, 'test', 'exports.js'), 'utf8')}`);
  try {
    await build({
      entryPoints: [testEntry],
      bundle: true,
      minify: false,
      format: 'esm',
      logLevel: 'warning',
      outfile: join(distDir, 'test', 'app.js'),
    });
  } finally {
    rmSync(testEntry, { force: true });
  }
  cpSync(join(root, 'extension', 'ui', 'dist', 'assets'), join(distDir, 'test', 'assets'), { recursive: true });
}

const artifactsDir = join(repoRoot, 'release-artifacts');
mkdirSync(artifactsDir, { recursive: true });
const zipPath = join(artifactsDir, `veles-tools-optimizer-${version}.zip`);
if (existsSync(zipPath)) {
  rmSync(zipPath);
}
const zip = spawnSync('zip', ['-qr', zipPath, '.', '-x', '*.DS_Store'], { cwd: extensionOut, stdio: 'inherit' });
if (zip.status !== 0) {
  throw new Error('zip failed');
}
const bundleBytes = readFileSync(join(extensionOut, 'ui', 'dist', 'assets', bundleName)).length;
console.log(`Built ${version}: ${bundleName} (${(bundleBytes / 1024).toFixed(0)} KB) → ${zipPath}`);

const fs = require('fs');
const path = require('path');
const readline = require('readline');
const babel = require('@babel/core');
const tf = require('@tensorflow/tfjs');

const repo = path.resolve(__dirname, '..');
const source = path.resolve(process.env.CHESS_SOURCE || repo);
const originalLoader = require.extensions['.js'];
require.extensions['.js'] = (module, filename) => {
  if (!filename.startsWith(path.join(source, 'src') + path.sep)) {
    return originalLoader(module, filename);
  }
  const { code } = babel.transformFileSync(filename, {
    babelrc: false,
    configFile: false,
    plugins: [require.resolve('@babel/plugin-transform-modules-commonjs')],
  });
  module._compile(code, filename);
};

tf.io.registerLoadRouter((url) => {
  if (typeof url !== 'string' || !url.endsWith('/nn/model.json')) return null;
  return { load: async () => {
    const directory = path.resolve(process.env.CHESS_MODEL || path.join(repo, 'public/nn'));
    const manifest = JSON.parse(fs.readFileSync(path.join(directory, 'model.json'), 'utf8'));
    const data = Buffer.concat(manifest.weightsManifest.flatMap((group) => (
      group.paths.map((file) => fs.readFileSync(path.join(directory, file)))
    )));
    return {
      modelTopology: manifest.modelTopology,
      weightSpecs: manifest.weightsManifest.flatMap((group) => group.weights),
      weightData: data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
    };
  } };
});

const { pickNNMove, warmNNModel } = require(path.join(source, 'src/ai/nnAI'));
const { pickMCTSMove } = require(path.join(source, 'src/ai/mctsAI'));
const types = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' };

function fromFen(fen) {
  const [placement, turn, rights, ep] = fen.split(' ');
  const pieces = {};
  placement.split('/').forEach((rank, x) => {
    let y = 0;
    for (const letter of rank) {
      if (/\d/.test(letter)) { y += Number(letter); continue; }
      const color = letter === letter.toUpperCase() ? 'white' : 'black';
      const type = types[letter.toLowerCase()];
      const home = color === 'white' ? 7 : 0;
      let hasMoved = x !== home;
      if (type === 'king') hasMoved = y !== 4 || x !== home || !(color === 'white' ? /[KQ]/ : /[kq]/).test(rights);
      if (type === 'rook') hasMoved = x !== home || !rights.includes(color === 'white' ? (y === 0 ? 'Q' : y === 7 ? 'K' : '!') : (y === 0 ? 'q' : y === 7 ? 'k' : '!'));
      pieces[`${x}-${y}`] = { color, type, hasMoved };
      y += 1;
    }
  });
  return {
    pieces,
    color: turn === 'w' ? 'white' : 'black',
    enPassantTarget: ep === '-' ? null : { x: 8 - Number(ep[1]), y: ep.charCodeAt(0) - 97 },
  };
}

function uci(move) {
  if (!move) return null;
  const square = ({ x, y }) => `${String.fromCharCode(97 + y)}${8 - x}`;
  return square(move.from) + square(move.to) + ({ queen: 'q', rook: 'r', bishop: 'b', knight: 'n' }[move.promotionType] || '');
}

async function main() {
  await tf.setBackend('cpu');
  await warmNNModel();
  const lines = readline.createInterface({ input: process.stdin });
  for await (const line of lines) {
    if (!line.trim()) continue;
    try {
      const request = JSON.parse(line);
      const { pieces, color, enPassantTarget } = fromFen(request.fen);
      const start = performance.now();
      let diagnostics = null;
      const options = { ...request.options, onSearchComplete: (result) => { diagnostics = result; } };
      const move = request.mode === 'mcts'
        ? await pickMCTSMove(pieces, color, enPassantTarget, options)
        : await pickNNMove(pieces, color, enPassantTarget, 2, options);
      console.log(JSON.stringify({ move: uci(move), elapsedMs: performance.now() - start, diagnostics }));
    } catch (error) {
      console.log(JSON.stringify({ error: error.stack }));
    }
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });

import { listLegalMoves, makeMove } from '../rules/chessRules';
import { scoreMove } from './chessHeuristics';
import { moveKey, moveToPolicyIndex, pickNNMove, POLICY_SIZE } from './nnAI';

const piece = (color, type, hasMoved = false) => ({ color, type, hasMoved });

function initialPieces() {
  const pieces = {};
  const backRank = ['rook', 'knight', 'bishop', 'queen', 'king', 'bishop', 'knight', 'rook'];
  backRank.forEach((type, column) => {
    pieces[`0-${column}`] = piece('black', type);
    pieces[`1-${column}`] = piece('black', 'pawn');
    pieces[`6-${column}`] = piece('white', 'pawn');
    pieces[`7-${column}`] = piece('white', type);
  });
  return pieces;
}

function play(pieces, from, to, enPassantTarget = null) {
  return makeMove(pieces, from, to, null, enPassantTarget);
}

describe('neural policy encoding', () => {
  test('promotion choices have distinct policy indices', () => {
    const base = {
      from: { x: 1, y: 0 },
      to: { x: 0, y: 0 },
      needsPromotion: true,
    };
    const indices = ['queen', 'rook', 'bishop', 'knight'].map((promotionType) =>
      moveToPolicyIndex({ ...base, promotionType }),
    );

    expect(new Set(indices).size).toBe(4);
    indices.forEach((index) => expect(index).toBeGreaterThanOrEqual(0));
    indices.forEach((index) => expect(index).toBeLessThan(POLICY_SIZE));
  });

  test('legacy policy mapping remains available during checkpoint migration', () => {
    const move = {
      from: { x: 6, y: 4 },
      to: { x: 4, y: 4 },
    };
    expect(moveToPolicyIndex(move, 4096)).toBe(12 * 64 + 28);
  });

  test('opening guard prefers development over weakening flank pawns', () => {
    const state = play(initialPieces(), { x: 6, y: 4 }, { x: 4, y: 4 });
    const moves = listLegalMoves(state.pieces, 'black', state.nextEnPassant);
    const findMove = (from, to) => moves.find((move) => (
      move.from.x === from.x && move.from.y === from.y &&
      move.to.x === to.x && move.to.y === to.y
    ));
    const knight = findMove({ x: 0, y: 6 }, { x: 2, y: 5 });
    const bPawn = findMove({ x: 1, y: 1 }, { x: 3, y: 1 });
    const fPawn = findMove({ x: 1, y: 5 }, { x: 2, y: 5 });

    expect(scoreMove(state.pieces, knight, 'black', state.nextEnPassant)).toBeGreaterThan(
      scoreMove(state.pieces, bPawn, 'black', state.nextEnPassant),
    );
    expect(scoreMove(state.pieces, knight, 'black', state.nextEnPassant)).toBeGreaterThan(
      scoreMove(state.pieces, fPawn, 'black', state.nextEnPassant),
    );
  });

  test('batches reply evaluation and rejects the live b-pawn regression', async () => {
    let state = play(initialPieces(), { x: 6, y: 4 }, { x: 4, y: 4 });
    state = play(state.pieces, { x: 0, y: 6 }, { x: 2, y: 5 }, state.nextEnPassant);
    state = play(state.pieces, { x: 6, y: 3 }, { x: 4, y: 3 }, state.nextEnPassant);

    const predictBatch = jest.fn(async (positions) => positions.map((position) => {
      const legalMoves = listLegalMoves(
        position.pieces,
        position.color,
        position.enPassantTarget,
      );
      const probability = 1 / Math.max(1, legalMoves.length);
      const priors = new Map(legalMoves.map((move) => [moveKey(move), probability]));
      const badMove = legalMoves.find((move) => (
        move.from.x === 1 && move.from.y === 1 && move.to.x === 3 && move.to.y === 1
      ));
      if (badMove) priors.set(moveKey(badMove), 0.9);
      return {
        value: 0,
        legalMoves,
        priors,
        neuralAvailable: true,
      };
    }));

    const chosen = await pickNNMove(
      state.pieces,
      'black',
      state.nextEnPassant,
      2,
      { predictBatch },
    );

    expect(chosen).not.toMatchObject({ from: { x: 1, y: 1 }, to: { x: 3, y: 1 } });
    expect(predictBatch).toHaveBeenCalledTimes(3);
    expect(predictBatch.mock.calls.some(([positions]) => positions.length > 1)).toBe(true);
  });

  test('plays an available checkmate without waiting for neural inference', async () => {
    const pieces = {
      '0-0': piece('black', 'king'),
      '2-2': piece('white', 'king'),
      '2-1': piece('white', 'queen'),
    };
    const predictBatch = jest.fn();

    const chosen = await pickNNMove(pieces, 'white', null, 2, { predictBatch });

    expect(chosen.from).toEqual({ x: 2, y: 1 });
    expect(chosen.to).toEqual({ x: 1, y: 1 });
    expect(predictBatch).not.toHaveBeenCalled();
  });
});

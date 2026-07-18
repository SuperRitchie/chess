import { listLegalMoves, makeMove } from '../rules/chessRules';

jest.mock('./nnAI', () => ({
  moveKey: (move) => `${move.from.x}-${move.from.y}:${move.to.x}-${move.to.y}:${move.promotionType || ''}`,
  predictPolicyValueBatchForPositions: jest.fn(),
  predictPolicyValueForMoves: jest.fn(),
}));

import {
  moveKey,
  predictPolicyValueBatchForPositions,
  predictPolicyValueForMoves,
} from './nnAI';
import { pickMCTSMove } from './mctsAI';

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

describe('neural PUCT MCTS', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    const predict = async (pieces, color, enPassantTarget) => {
      const legalMoves = listLegalMoves(pieces, color, enPassantTarget);
      const probability = 1 / Math.max(1, legalMoves.length);
      const priors = new Map(legalMoves.map((move) => [moveKey(move), probability]));
      return { value: 0, priors, legalMoves };
    };
    predictPolicyValueForMoves.mockImplementation(predict);
    predictPolicyValueBatchForPositions.mockImplementation(
      async (positions) => Promise.all(
        positions.map(({ pieces, color, enPassantTarget }) => predict(pieces, color, enPassantTarget)),
      ),
    );
  });

  test('returns one of the legal moves', async () => {
    const pieces = {
      '7-4': piece('white', 'king'),
      '6-4': piece('white', 'pawn'),
      '0-4': piece('black', 'king'),
      '1-4': piece('black', 'pawn'),
    };

    const legalMoves = listLegalMoves(pieces, 'white', null);
    const chosen = await pickMCTSMove(pieces, 'white', null, {
      timeMs: 1000,
      maxIterations: 8,
      batchSize: 4,
      cpuct: 1.5,
    });

    expect(predictPolicyValueBatchForPositions).toHaveBeenCalled();
    expect(
      predictPolicyValueBatchForPositions.mock.calls.some(([positions]) => positions.length > 1),
    ).toBe(true);
    expect(chosen).not.toBeNull();
    expect(
      legalMoves.some(
        (move) =>
          move.from.x === chosen.from.x &&
          move.from.y === chosen.from.y &&
          move.to.x === chosen.to.x &&
          move.to.y === chosen.to.y &&
          (move.promotionType || null) === (chosen.promotionType || null),
      ),
    ).toBe(true);
  });

  test('plays an available checkmate before neural search', async () => {
    const pieces = {
      '0-0': piece('black', 'king'),
      '2-2': piece('white', 'king'),
      '2-1': piece('white', 'queen'),
    };

    const chosen = await pickMCTSMove(pieces, 'white', null, {
      timeMs: 1000,
      maxIterations: 1,
      batchSize: 1,
    });

    expect(chosen.from).toEqual({ x: 2, y: 1 });
    expect(chosen.to).toEqual({ x: 1, y: 1 });
    expect(predictPolicyValueBatchForPositions).not.toHaveBeenCalled();
  });

  test('does not repeat the live early f-pawn regression', async () => {
    let state = makeMove(initialPieces(), { x: 6, y: 4 }, { x: 4, y: 4 });
    state = makeMove(state.pieces, { x: 1, y: 2 }, { x: 2, y: 2 }, null, state.nextEnPassant);
    state = makeMove(state.pieces, { x: 6, y: 3 }, { x: 4, y: 3 }, null, state.nextEnPassant);

    const chosen = await pickMCTSMove(state.pieces, 'black', state.nextEnPassant, {
      timeMs: 1000,
      maxIterations: 96,
      batchSize: 8,
    });

    expect(chosen).not.toMatchObject({ from: { x: 1, y: 5 }, to: { x: 2, y: 5 } });
  });
});

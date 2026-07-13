import { listLegalMoves } from '../rules/chessRules';

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

describe('neural PUCT MCTS', () => {
  beforeEach(() => {
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
});

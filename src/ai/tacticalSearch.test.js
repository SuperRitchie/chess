import { listLegalMoves } from '../rules/chessRules';
import { findMateInOne, searchMoveKey, searchTacticalMoves } from './tacticalSearch';

const piece = (color, type) => ({ color, type, hasMoved: true });

describe('complete tactical verification', () => {
  test('finds mate even when the policy gives it no probability', () => {
    const pieces = {
      '0-0': piece('black', 'king'),
      '2-2': piece('white', 'king'),
      '2-1': piece('white', 'queen'),
    };
    const mate = findMateInOne(pieces, 'white', null);
    expect(mate.to).toEqual({ x: 1, y: 1 });
  });

  test('rejects a queen capture that loses to a rook despite a defending pawn', () => {
    const pieces = {
      '7-6': piece('white', 'king'),
      '7-3': piece('white', 'queen'),
      '4-4': piece('white', 'pawn'),
      '0-6': piece('black', 'king'),
      '0-3': piece('black', 'rook'),
      '3-3': piece('black', 'pawn'),
    };
    const moves = listLegalMoves(pieces, 'white');
    const capture = moves.find((move) => move.to.x === 3 && move.to.y === 3);
    const result = searchTacticalMoves(pieces, 'white', null, {
      neuralScores: new Map([[searchMoveKey(capture), 1]]),
      timeMs: 10000, maxDepth: 2,
    });
    expect(result.depth).toBe(2);
    expect(result.move).not.toMatchObject({ from: { x: 7, y: 3 }, to: { x: 3, y: 3 } });
  });

  test('keeps the last complete root iteration when its node budget runs out', () => {
    const pieces = {
      '7-6': piece('white', 'king'),
      '7-0': piece('white', 'rook'),
      '0-6': piece('black', 'king'),
      '0-0': piece('black', 'queen'),
    };
    const completed = searchTacticalMoves(pieces, 'white', null, { timeMs: 10000, maxDepth: 1 });
    const interrupted = searchTacticalMoves(pieces, 'white', null, {
      timeMs: 10000, maxDepth: 3, maxNodes: completed.nodes + 1,
    });
    expect(interrupted.depth).toBe(1);
    expect(interrupted.move).toEqual(completed.move);
    expect(interrupted.move.to).toEqual({ x: 0, y: 0 });
  });
});

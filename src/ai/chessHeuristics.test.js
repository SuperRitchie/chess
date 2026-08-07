import { listLegalMoves } from '../rules/chessRules';
import {
  evaluatePosition,
  orderMoves,
  positionKey,
  tacticalScoreForMove,
} from './chessHeuristics';

const piece = (color, type, hasMoved = false) => ({ color, type, hasMoved });

describe('phase-aware chess search heuristics', () => {
  test('values a passed pawn above the same pawn when blocked', () => {
    const common = {
      '6-7': piece('white', 'king'),
      '0-7': piece('black', 'king'),
      '2-3': piece('white', 'pawn', true),
    };
    const passed = { ...common, '1-0': piece('black', 'pawn') };
    const blocked = { ...common, '1-3': piece('black', 'pawn') };

    expect(evaluatePosition(passed, 'white')).toBeGreaterThan(
      evaluatePosition(blocked, 'white'),
    );
  });

  test('activates the king in the endgame', () => {
    const centralKing = {
      '4-4': piece('white', 'king', true),
      '0-0': piece('black', 'king', true),
    };
    const cornerKing = {
      '7-7': piece('white', 'king', true),
      '0-0': piece('black', 'king', true),
    };

    expect(evaluatePosition(centralKing, 'white')).toBeGreaterThan(
      evaluatePosition(cornerKing, 'white'),
    );
  });

  test('orders a free queen capture first and verifies it tactically', () => {
    const pieces = {
      '6-7': piece('white', 'king'),
      '7-0': piece('white', 'rook'),
      '0-4': piece('black', 'king'),
      '0-0': piece('black', 'queen'),
    };
    const moves = listLegalMoves(pieces, 'white', null);
    const capture = moves.find((move) => (
      move.from.x === 7 && move.from.y === 0 && move.to.x === 0 && move.to.y === 0
    ));
    const quiet = moves.find((move) => (
      move.from.x === 7 && move.from.y === 0 && move.to.x === 6 && move.to.y === 0
    ));
    const ordered = orderMoves(pieces, moves, 'white', null);

    expect(ordered[0].move).toMatchObject(capture);
    expect(tacticalScoreForMove(pieces, capture, 'white')).toBeGreaterThan(
      tacticalScoreForMove(pieces, quiet, 'white'),
    );
  });

  test('keeps distinct piece types distinct in transposition keys', () => {
    const knight = { '7-4': piece('white', 'king'), '0-0': piece('black', 'knight') };
    const king = { '7-4': piece('white', 'king'), '0-0': piece('black', 'king') };

    expect(positionKey(knight, 'white')).not.toBe(positionKey(king, 'white'));
  });
});

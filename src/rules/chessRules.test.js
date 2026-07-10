import { isLegalMove, listLegalMoves } from './chessRules';

const piece = (color, type, hasMoved = false) => ({ color, type, hasMoved });

describe('chess rules', () => {
  test('AI move generation includes every promotion piece', () => {
    const pieces = {
      '1-0': piece('white', 'pawn', true),
      '7-7': piece('white', 'king'),
      '0-7': piece('black', 'king'),
    };

    const promotions = listLegalMoves(pieces, 'white').filter(
      (move) => move.from.x === 1 && move.from.y === 0 && move.to.x === 0 && move.to.y === 0,
    );

    expect(promotions.map((move) => move.promotionType).sort()).toEqual(
      ['bishop', 'knight', 'queen', 'rook'],
    );
  });

  test('castling is rejected when moving the king exposes an x-ray attack', () => {
    const pieces = {
      '7-4': piece('white', 'king'),
      '7-0': piece('white', 'rook'),
      '0-0': piece('black', 'king'),
      '7-7': piece('black', 'rook'),
    };

    expect(isLegalMove(pieces, { x: 7, y: 4 }, { x: 7, y: 2 }, true, null)).toBe(false);
  });
});

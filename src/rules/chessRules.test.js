import { isLegalMove, listLegalMoves, makeMove } from './chessRules';

const piece = (color, type, hasMoved = false) => ({ color, type, hasMoved });

function fromFen(fen) {
  const pieces = {};
  const types = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' };
  fen.split(' ')[0].split('/').forEach((rank, x) => {
    let y = 0;
    for (const letter of rank) {
      if (/\d/.test(letter)) y += Number(letter);
      else { pieces[`${x}-${y}`] = piece(letter === letter.toUpperCase() ? 'white' : 'black', types[letter.toLowerCase()]); y += 1; }
    }
  });
  return pieces;
}

function perft(pieces, color, ep, depth) {
  if (depth === 0) return 1;
  return listLegalMoves(pieces, color, ep).reduce((total, move) => {
    const next = makeMove(pieces, move.from, move.to, move.promotionType, ep);
    return total + perft(next.pieces, color === 'white' ? 'black' : 'white', next.nextEnPassant, depth - 1);
  }, 0);
}

describe('chess rules', () => {
  test('matches all 8902 legal opening continuations through three plies', () => {
    const pieces = fromFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR');
    expect(perft(pieces, 'white', null, 1)).toBe(20);
    expect(perft(pieces, 'white', null, 2)).toBe(400);
    expect(perft(pieces, 'white', null, 3)).toBe(8902);
  });

  test('matches castling and pinned-piece continuations in Kiwipete', () => {
    const pieces = fromFen('r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R');
    expect(perft(pieces, 'white', null, 1)).toBe(48);
    expect(perft(pieces, 'white', null, 2)).toBe(2039);
  });

  test('marks en passant as a capture and rejects an exposed king', () => {
    const pieces = fromFen('4k3/8/8/3pP3/8/8/8/4K3');
    const ep = { x: 2, y: 3 };
    expect(listLegalMoves(pieces, 'white', ep)).toContainEqual({
      from: { x: 3, y: 4 }, to: { x: 2, y: 3 }, capture: true, needsPromotion: false,
    });
    const pinned = fromFen('k3r3/8/8/3pP3/8/8/8/4K3');
    expect(isLegalMove(pinned, { x: 3, y: 4 }, { x: 2, y: 3 }, true, ep)).toBe(false);
  });
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

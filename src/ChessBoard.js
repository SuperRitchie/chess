// src/ChessBoard.jsx
import React, { useMemo, useState } from 'react';
import './index.css';
import ChessPiece from './ChessPiece';
import move_sound from './assets/move.mp3';
import check from './assets/check.mp3';
import { isLegalMove, makeMove, getPiece, isKingInCheck, hasAnyLegalMove } from './chessRules';

const initPiece = (color, type) => ({ color, type, hasMoved: false });

// --- Algebraic helpers ---
const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
const fileOf = (y) => files[y];
const rankOf = (x) => String(8 - x);
const sq = (x, y) => `${fileOf(y)}${rankOf(x)}`;

export default function ChessBoard() {
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [isWhiteTurn, setIsWhiteTurn] = useState(true);
  const [legalMoves, setLegalMoves] = useState([]); // [{x,y,capture:boolean}]
  const [promotionPending, setPromotionPending] = useState(null); // {from,to,color,nextTurnIsWhite,prevPieces,prevEnPassant}
  const [frozenForPromotion, setFrozenForPromotion] = useState(false);
  const [enPassantTarget, setEnPassantTarget] = useState(null);   // {x,y} or null
  const [movesSAN, setMovesSAN] = useState([]); // flat list: ["e4","e5","Nf3",...]
  const [pieces, setPieces] = useState({
    '0-0': initPiece('black', 'rook'),
    '0-1': initPiece('black', 'knight'),
    '0-2': initPiece('black', 'bishop'),
    '0-3': initPiece('black', 'queen'),
    '0-4': initPiece('black', 'king'),
    '0-5': initPiece('black', 'bishop'),
    '0-6': initPiece('black', 'knight'),
    '0-7': initPiece('black', 'rook'),
    '1-0': initPiece('black', 'pawn'),
    '1-1': initPiece('black', 'pawn'),
    '1-2': initPiece('black', 'pawn'),
    '1-3': initPiece('black', 'pawn'),
    '1-4': initPiece('black', 'pawn'),
    '1-5': initPiece('black', 'pawn'),
    '1-6': initPiece('black', 'pawn'),
    '1-7': initPiece('black', 'pawn'),
    '6-0': initPiece('white', 'pawn'),
    '6-1': initPiece('white', 'pawn'),
    '6-2': initPiece('white', 'pawn'),
    '6-3': initPiece('white', 'pawn'),
    '6-4': initPiece('white', 'pawn'),
    '6-5': initPiece('white', 'pawn'),
    '6-6': initPiece('white', 'pawn'),
    '6-7': initPiece('white', 'pawn'),
    '7-0': initPiece('white', 'rook'),
    '7-1': initPiece('white', 'knight'),
    '7-2': initPiece('white', 'bishop'),
    '7-3': initPiece('white', 'queen'),
    '7-4': initPiece('white', 'king'),
    '7-5': initPiece('white', 'bishop'),
    '7-6': initPiece('white', 'knight'),
    '7-7': initPiece('white', 'rook'),
  });

  // --- UI bits ---
  const sideToMoveLabel = isWhiteTurn ? 'White' : 'Black';

  // --- Move highlighting ---
  const computeLegalMoves = (fromX, fromY) => {
    const res = [];
    for (let ty = 0; ty < 8; ty++) {
      for (let tx = 0; tx < 8; tx++) {
        const from = { x: fromX, y: fromY };
        const to = { x: tx, y: ty };
        if (isLegalMove(pieces, from, to, isWhiteTurn, enPassantTarget)) {
          const targetPiece = getPiece(pieces, tx, ty);
          const capture = !!targetPiece || willBeEnPassant(from, to);
          res.push({ x: tx, y: ty, capture });
        }
      }
    }
    return res;
  };
  const legalForSquare = (x, y) => legalMoves.find(m => m.x === x && m.y === y);
  const clearSelection = () => { setSelectedSquare(null); setLegalMoves([]); };

  const playSound = (audioFile) => { new Audio(audioFile).play(); };

  const willBeEnPassant = (from, to) => {
    const mover = getPiece(pieces, from.x, from.y);
    if (!mover || mover.type !== 'pawn' || !enPassantTarget) return false;
    const dx = to.x - from.x, dy = to.y - from.y;
    const destEmpty = !getPiece(pieces, to.x, to.y);
    const dir = mover.color === 'white' ? -1 : 1;
    return destEmpty && dx === dir && Math.abs(dy) === 1 &&
      enPassantTarget.x === to.x && enPassantTarget.y === to.y;
  };

  // --- SAN/PGN generation ---
  const pieceLetter = { pawn: '', knight: 'N', bishop: 'B', rook: 'R', queen: 'Q', king: 'K' };

  function otherSameTypeCanAlsoGo(prevPieces, color, type, from, to) {
    // Check if another same-type piece (same color) can legally go to `to`
    for (let x = 0; x < 8; x++) {
      for (let y = 0; y < 8; y++) {
        if (x === from.x && y === from.y) continue;
        const p = getPiece(prevPieces, x, y);
        if (!p || p.color !== color || p.type !== type) continue;
        if (isLegalMove(prevPieces, { x, y }, to, color === 'white', enPassantTarget)) {
          return { x, y };
        }
      }
    }
    return null;
  }

  function isCastle(from, to, mover) {
    return mover.type === 'king' && from.x === to.x && Math.abs(to.y - from.y) === 2;
  }

  function genSAN(prevPieces, afterPieces, from, to, moverBefore, promotionType, newEnPassantTarget) {
    const moving = moverBefore; // piece before move
    const color = moving.color;
    const enemy = color === 'white' ? 'black' : 'white';
    const destBefore = getPiece(prevPieces, to.x, to.y);
    const enPassant = (moving.type === 'pawn' && !destBefore && willBeEnPassant(from, to));
    const wasCapture = !!destBefore || enPassant;

    // Castling first
    if (isCastle(from, to, moving)) {
      let san = (to.y > from.y) ? 'O-O' : 'O-O-O';
      // check/mate suffix
      const opponentToMove = enemy;
      const inCheck = isKingInCheck(afterPieces, opponentToMove);
      const anyMoves = hasAnyLegalMove(afterPieces, opponentToMove, newEnPassantTarget);
      if (inCheck) san += anyMoves ? '+' : '#';
      return san;
    }

    let san = '';
    if (moving.type === 'pawn') {
      if (wasCapture) san += fileOf(from.y) + 'x';
      san += sq(to.x, to.y);
      if (promotionType) san += '=' + pieceLetter[promotionType];
    } else {
      san += pieceLetter[moving.type];

      // Disambiguation if needed
      const conflict = otherSameTypeCanAlsoGo(prevPieces, color, moving.type, from, to);
      if (conflict) {
        const needFile = conflict.y !== from.y;
        const needRank = conflict.x !== from.x;
        if (needFile) san += fileOf(from.y);
        if (!needFile && needRank) san += rankOf(from.x);
        if (needFile && needRank) san += rankOf(from.x); // full if both differ
      }

      if (wasCapture) san += 'x';
      san += sq(to.x, to.y);
    }

    // Check / Checkmate markers
    const opponentToMove = enemy;
    const inCheck = isKingInCheck(afterPieces, opponentToMove);
    const anyMoves = hasAnyLegalMove(afterPieces, opponentToMove, newEnPassantTarget);
    if (inCheck) san += anyMoves ? '+' : '#';

    return san;
  }

  const pgnText = useMemo(() => {
    // Build "1. e4 e5 2. Nf3 Nc6 ..." from flat SAN array
    const parts = [];
    for (let i = 0; i < movesSAN.length; i += 2) {
      const moveNo = (i / 2) + 1;
      const whiteMove = movesSAN[i] ?? '';
      const blackMove = movesSAN[i + 1] ?? '';
      if (blackMove) {
        parts.push(`${moveNo}. ${whiteMove} ${blackMove}`);
      } else {
        parts.push(`${moveNo}. ${whiteMove}`);
      }
    }
    return parts.join(' ');
  }, [movesSAN]);

  /** Evaluate state and (optionally) alert on mate/stalemate */
  const evaluatePostMoveState = (newPieces, nextTurnIsWhite, nextEnPassant) => {
    const sideToMove = nextTurnIsWhite ? 'white' : 'black';
    const inCheck = isKingInCheck(newPieces, sideToMove);
    const anyMoves = hasAnyLegalMove(newPieces, sideToMove, nextEnPassant);

    if (inCheck) playSound(check);
    if (!anyMoves && inCheck) {
      alert(`${sideToMove} is checkmated!`);
    } else if (!anyMoves && !inCheck) {
      alert('Stalemate!');
    }
  };

  // --- Promotion helpers (now includes SAN after choice) ---
  const maybeStartPromotion = (beforePieces, from, to, nextTurnIsWhite, prevEnPassant) => {
    const movedAfter = getPiece(beforePieces, to.x, to.y);
    if (!movedAfter || movedAfter.type !== 'pawn') return false;
    const promoteRow = movedAfter.color === 'white' ? 0 : 7;
    if (to.x !== promoteRow) return false;

    setPromotionPending({ from, to, color: movedAfter.color, nextTurnIsWhite, prevPieces: beforePieces, prevEnPassant });
    setFrozenForPromotion(true);
    return true;
  };

  const finishPromotion = (choiceType) => {
    if (!promotionPending) return;
    const { from, to, color, nextTurnIsWhite, prevPieces } = promotionPending;

    // Replace the pawn with chosen piece
    const k = `${to.x}-${to.y}`;
    const updated = {
      ...pieces,
      [k]: { color, type: choiceType, hasMoved: true }
    };

    // Generate SAN for the promotion move
    const moverBefore = getPiece(prevPieces, from.x, from.y);
    const newEnPassantTarget = null; // promotion cannot follow a double push; already handled earlier
    const san = genSAN(prevPieces, updated, from, to, moverBefore, choiceType, newEnPassantTarget);
    setMovesSAN((arr) => [...arr, san]);

    setPieces(updated);
    setPromotionPending(null);
    setFrozenForPromotion(false);
    setIsWhiteTurn(nextTurnIsWhite);
    evaluatePostMoveState(updated, nextTurnIsWhite, newEnPassantTarget);
  };

  // --- Input handlers ---
  const handleSquareClick = (x, y) => {
    if (frozenForPromotion) return;
    const piece = getPiece(pieces, x, y);

    if (selectedSquare) {
      const from = { x: selectedSquare.x, y: selectedSquare.y };
      const to = { x, y };

      if (isLegalMove(pieces, from, to, isWhiteTurn, enPassantTarget)) {
        const prevPieces = pieces;
        const moverBefore = getPiece(prevPieces, from.x, from.y);
        const { pieces: basePieces, nextEnPassant } = makeMove(prevPieces, from, to, null, enPassantTarget);
        playSound(move_sound);
        clearSelection();

        // Promotion path pauses SAN until user chooses piece
        const nextTurnIsWhite = !isWhiteTurn;
        setPieces(basePieces);
        setEnPassantTarget(nextEnPassant);

        if (maybeStartPromotion(basePieces, from, to, nextTurnIsWhite, enPassantTarget)) {
          return;
        }

        // Normal move SAN (no promotion)
        const san = genSAN(prevPieces, basePieces, from, to, moverBefore, null, nextEnPassant);
        setMovesSAN((arr) => [...arr, san]);

        setIsWhiteTurn(nextTurnIsWhite);
        evaluatePostMoveState(basePieces, nextTurnIsWhite, nextEnPassant);
        return;
      }

      if (selectedSquare.x === x && selectedSquare.y === y) { clearSelection(); return; }
      if (piece && ((isWhiteTurn && piece.color === 'white') || (!isWhiteTurn && piece.color === 'black'))) {
        setSelectedSquare({ x, y });
        setLegalMoves(computeLegalMoves(x, y));
      } else {
        playSound(check);
        clearSelection();
      }
      return;
    }

    if (piece && ((isWhiteTurn && piece.color === 'white') || (!isWhiteTurn && piece.color === 'black'))) {
      setSelectedSquare({ x, y });
      setLegalMoves(computeLegalMoves(x, y));
    }
  };

  const handleOnDrag = (e, x, y) => {
    if (frozenForPromotion) { e.preventDefault(); return; }
    const piece = getPiece(pieces, x, y);
    if (!piece) { e.preventDefault(); return; }
    if ((isWhiteTurn && piece.color !== 'white') || (!isWhiteTurn && piece.color !== 'black')) {
      e.preventDefault(); return;
    }
    setSelectedSquare({ x, y });
    setLegalMoves(computeLegalMoves(x, y));
    e.dataTransfer.setData('position', `${x}-${y}`);
  };

  const handleDragOver = (e) => e.preventDefault();

  const handleOnDrop = (e, x, y) => {
    if (frozenForPromotion) return;
    const data = e.dataTransfer.getData('position');
    if (!data) return;
    const [oldX, oldY] = data.split('-').map(Number);
    const from = { x: oldX, y: oldY };
    const to = { x, y };

    if (!isLegalMove(pieces, from, to, isWhiteTurn, enPassantTarget)) {
      playSound(check);
      return;
    }

    const prevPieces = pieces;
    const moverBefore = getPiece(prevPieces, from.x, from.y);
    const { pieces: basePieces, nextEnPassant } = makeMove(prevPieces, from, to, null, enPassantTarget);
    playSound(move_sound);
    clearSelection();

    const nextTurnIsWhite = !isWhiteTurn;
    setPieces(basePieces);
    setEnPassantTarget(nextEnPassant);

    if (maybeStartPromotion(basePieces, from, to, nextTurnIsWhite, enPassantTarget)) {
      return;
    }

    const san = genSAN(prevPieces, basePieces, from, to, moverBefore, null, nextEnPassant);
    setMovesSAN((arr) => [...arr, san]);

    setIsWhiteTurn(nextTurnIsWhite);
    evaluatePostMoveState(basePieces, nextTurnIsWhite, nextEnPassant);
  };

  // --- Render ---
  return (
    <div className="chess-layout">
      <div className="chess-sidepanel">
        <div className="turn-indicator"><b>Turn:</b> {sideToMoveLabel}</div>
        <div className="pgn-title">PGN</div>
        <div className="pgn-box">{pgnText || '—'}</div>
      </div>

      {/* Promotion modal */}
      {promotionPending && (
        <div className="promotion-modal">
          <div className="promotion-card">
            <div className="promotion-title">Choose promotion</div>
            <div className="promotion-choices">
              <button onClick={() => finishPromotion('queen')}>Queen</button>
              <button onClick={() => finishPromotion('rook')}>Rook</button>
              <button onClick={() => finishPromotion('bishop')}>Bishop</button>
              <button onClick={() => finishPromotion('knight')}>Knight</button>
            </div>
          </div>
        </div>
      )}

      <div className="chessboard">
        {Array.from({ length: 8 }, (_, y) => (
          <div key={y} className="chessboard-row">
            {Array.from({ length: 8 }, (_, x) => {
              const piece = getPiece(pieces, x, y);
              const selected = selectedSquare && selectedSquare.x === x && selectedSquare.y === y;
              const lm = legalForSquare(x, y);
              const isCapture = lm?.capture;

              return (
                <div
                  key={`${x}-${y}`}
                  className={[
                    'chessboard-square',
                    ((x + y) % 2 === 1) ? 'chessboard-square--black' : 'chessboard-square--white',
                    selected ? 'chessboard-square--selected' : '',
                    lm ? (isCapture ? 'chessboard-square--legal-capture' : 'chessboard-square--legal-move') : '',
                    frozenForPromotion ? 'chessboard-square--disabled' : '',
                  ].join(' ')}
                  onClick={() => handleSquareClick(x, y)}
                  onDrop={(e) => handleOnDrop(e, x, y)}
                  onDragOver={handleDragOver}
                >
                  <div
                    className="chesspiece-wrapper"
                    draggable={
                      !!piece &&
                      !frozenForPromotion &&
                      ((isWhiteTurn && piece.color === 'white') || (!isWhiteTurn && piece.color === 'black'))
                    }
                    onDragStart={(e) => handleOnDrag(e, x, y)}
                  >
                    <ChessPiece piece={piece} />
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

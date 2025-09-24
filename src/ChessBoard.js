// src/ChessBoard.jsx
import React, { useState } from 'react';
import './index.css';
import ChessPiece from './ChessPiece';
import move_sound from './assets/move.mp3';
import check from './assets/check.mp3';
import { isLegalMove, makeMove, getPiece, isKingInCheck, hasAnyLegalMove } from './chessRules';

const initPiece = (color, type) => ({ color, type, hasMoved: false });

const ChessBoard = () => {
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [isWhiteTurn, setIsWhiteTurn] = useState(true);
  const [legalMoves, setLegalMoves] = useState([]); // [{x,y,capture:boolean}]
  const [promotionPending, setPromotionPending] = useState(null); // {to:{x,y}, color, nextTurnIsWhite}
  const [frozenForPromotion, setFrozenForPromotion] = useState(false);
  const [enPassantTarget, setEnPassantTarget] = useState(null);   // {x,y} or null

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

  const computeLegalMoves = (fromX, fromY) => {
    const res = [];
    for (let ty = 0; ty < 8; ty++) {
      for (let tx = 0; tx < 8; tx++) {
        const from = { x: fromX, y: fromY };
        const to = { x: tx, y: ty };
        if (isLegalMove(pieces, from, to, isWhiteTurn, enPassantTarget)) {
          const targetPiece = getPiece(pieces, tx, ty);
          // For en passant, the destination is empty; still show as capture
          const capture = !!targetPiece || willBeEnPassant(from, to);
          res.push({ x: tx, y: ty, capture });
        }
      }
    }
    return res;
  };

  const willBeEnPassant = (from, to) => {
    const mover = getPiece(pieces, from.x, from.y);
    if (!mover || mover.type !== 'pawn' || !enPassantTarget) return false;
    const dx = to.x - from.x, dy = to.y - from.y;
    const destEmpty = !getPiece(pieces, to.x, to.y);
    const dir = mover.color === 'white' ? -1 : 1;
    return destEmpty && dx === dir && Math.abs(dy) === 1 &&
      enPassantTarget.x === to.x && enPassantTarget.y === to.y;
  };

  const clearSelection = () => {
    setSelectedSquare(null);
    setLegalMoves([]);
  };

  const playSound = (audioFile) => {
    const audio = new Audio(audioFile);
    audio.play();
  };

  const legalForSquare = (x, y) => legalMoves.find(m => m.x === x && m.y === y);

  /** After a successful move, check for check / checkmate / stalemate */
  const evaluatePostMoveState = (newPieces, nextTurnIsWhite) => {
    const sideToMove = nextTurnIsWhite ? 'white' : 'black';
    const inCheck = isKingInCheck(newPieces, sideToMove);
    const anyMoves = hasAnyLegalMove(newPieces, sideToMove, enPassantTarget /* old one doesn't matter after move */);

    if (inCheck) playSound(check);

    if (!anyMoves && inCheck) {
      alert(`${sideToMove} is checkmated!`);
    } else if (!anyMoves && !inCheck) {
      alert('Stalemate!');
    }
  };

  // --- Promotion helpers ---
  const maybeStartPromotion = (beforePieces, from, to, nextTurnIsWhite) => {
    const moved = getPiece(beforePieces, to.x, to.y);
    if (!moved || moved.type !== 'pawn') return false;
    const promoteRow = moved.color === 'white' ? 0 : 7;
    if (to.x !== promoteRow) return false;

    setPromotionPending({ to, color: moved.color, nextTurnIsWhite });
    setFrozenForPromotion(true);
    return true;
  };

  const finishPromotion = (choiceType) => {
    if (!promotionPending) return;

    const { to, color, nextTurnIsWhite } = promotionPending;
    const k = `${to.x}-${to.y}`;
    const updated = {
      ...pieces,
      [k]: { color, type: choiceType, hasMoved: true }
    };

    setPieces(updated);
    setPromotionPending(null);
    setFrozenForPromotion(false);
    setIsWhiteTurn(nextTurnIsWhite);
    evaluatePostMoveState(updated, nextTurnIsWhite);
  };

  // CLICK-TO-MOVE logic
  const handleSquareClick = (x, y) => {
    if (frozenForPromotion) return;
    const piece = getPiece(pieces, x, y);

    if (selectedSquare) {
      const from = { x: selectedSquare.x, y: selectedSquare.y };
      const to = { x, y };

      if (isLegalMove(pieces, from, to, isWhiteTurn, enPassantTarget)) {
        const { pieces: basePieces, nextEnPassant } = makeMove(pieces, from, to, null, enPassantTarget);
        playSound(move_sound);
        clearSelection();

        const nextTurnIsWhite = !isWhiteTurn;
        setPieces(basePieces);
        setEnPassantTarget(nextEnPassant);

        if (maybeStartPromotion(basePieces, from, to, nextTurnIsWhite)) {
          // Pause turn switch until promotion selection
          return;
        }

        setIsWhiteTurn(nextTurnIsWhite);
        evaluatePostMoveState(basePieces, nextTurnIsWhite);
        return;
      }

      if (selectedSquare.x === x && selectedSquare.y === y) {
        clearSelection();
        return;
      }

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

  // Drag support
  const handleOnDrag = (e, x, y) => {
    if (frozenForPromotion) { e.preventDefault(); return; }
    const piece = getPiece(pieces, x, y);
    if (!piece) { e.preventDefault(); return; }
    if ((isWhiteTurn && piece.color !== 'white') || (!isWhiteTurn && piece.color !== 'black')) {
      e.preventDefault();
      return;
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

    const { pieces: basePieces, nextEnPassant } = makeMove(pieces, from, to, null, enPassantTarget);
    playSound(move_sound);
    clearSelection();

    const nextTurnIsWhite = !isWhiteTurn;
    setPieces(basePieces);
    setEnPassantTarget(nextEnPassant);

    if (maybeStartPromotion(basePieces, from, to, nextTurnIsWhite)) {
      return;
    }

    setIsWhiteTurn(nextTurnIsWhite);
    evaluatePostMoveState(basePieces, nextTurnIsWhite);
  };

  return (
    <div className="chessboard-wrapper">
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
};

export default ChessBoard;

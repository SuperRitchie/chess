// src/ChessBoard.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import "./index.css";
import ChessPiece from "./ChessPiece";
import move_sound from "./assets/move.mp3";
import check from "./assets/check.mp3";
import {
  isLegalMove,
  makeMove,
  getPiece,
  isKingInCheck,
  hasAnyLegalMove,
} from "./chessRules";
import { pickRandomMove } from "./randomAI";
import { pickMCTSMove } from "./mctsAI";
import { pickNNMove } from "./nnAI";

const initPiece = (color, type) => ({ color, type, hasMoved: false });

// --- Algebraic helpers for SAN/PGN ---
const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
const fileOf = (y) => files[y];
const rankOf = (x) => String(8 - x);
const sq = (x, y) => `${fileOf(y)}${rankOf(x)}`;
const pieceLetter = {
  pawn: "",
  knight: "N",
  bishop: "B",
  rook: "R",
  queen: "Q",
  king: "K",
};
const enemy = (c) => (c === "white" ? "black" : "white");

function makeInitialState() {
  return {
    selectedSquare: null,
    isWhiteTurn: true,
    legalMoves: [],
    promotionPending: null,
    frozenForPromotion: false,
    enPassantTarget: null,
    movesSAN: [],
    pieces: {
      "0-0": initPiece("black", "rook"),
      "0-1": initPiece("black", "knight"),
      "0-2": initPiece("black", "bishop"),
      "0-3": initPiece("black", "king"),
      "0-4": initPiece("black", "queen"),
      "0-5": initPiece("black", "bishop"),
      "0-6": initPiece("black", "knight"),
      "0-7": initPiece("black", "rook"),
      "1-0": initPiece("black", "pawn"),
      "1-1": initPiece("black", "pawn"),
      "1-2": initPiece("black", "pawn"),
      "1-3": initPiece("black", "pawn"),
      "1-4": initPiece("black", "pawn"),
      "1-5": initPiece("black", "pawn"),
      "1-6": initPiece("black", "pawn"),
      "1-7": initPiece("black", "pawn"),
      "6-0": initPiece("white", "pawn"),
      "6-1": initPiece("white", "pawn"),
      "6-2": initPiece("white", "pawn"),
      "6-3": initPiece("white", "pawn"),
      "6-4": initPiece("white", "pawn"),
      "6-5": initPiece("white", "pawn"),
      "6-6": initPiece("white", "pawn"),
      "6-7": initPiece("white", "pawn"),
      "7-0": initPiece("white", "rook"),
      "7-1": initPiece("white", "knight"),
      "7-2": initPiece("white", "bishop"),
      "7-3": initPiece("white", "king"),
      "7-4": initPiece("white", "queen"),
      "7-5": initPiece("white", "bishop"),
      "7-6": initPiece("white", "knight"),
      "7-7": initPiece("white", "rook"),
    },
  };
}

export default function ChessBoard() {
  const initial = useRef(makeInitialState());

  const [selectedSquare, setSelectedSquare] = useState(initial.current.selectedSquare);
  const [isWhiteTurn, setIsWhiteTurn] = useState(initial.current.isWhiteTurn);
  const [legalMoves, setLegalMoves] = useState(initial.current.legalMoves);
  const [promotionPending, setPromotionPending] = useState(initial.current.promotionPending);
  const [frozenForPromotion, setFrozenForPromotion] = useState(initial.current.frozenForPromotion);
  const [enPassantTarget, setEnPassantTarget] = useState(initial.current.enPassantTarget);
  const [movesSAN, setMovesSAN] = useState(initial.current.movesSAN);
  const [pieces, setPieces] = useState(initial.current.pieces);

  // modes
  const [whiteMode, setWhiteMode] = useState("human"); // 'human' | 'random' | 'mcts' | 'nn'
  const [blackMode, setBlackMode] = useState("human");
  const [isThinking, setIsThinking] = useState(false);

  // board UI
  const [isFlipped, setIsFlipped] = useState(false);

  // undo history (store previous snapshots)
  const [history, setHistory] = useState([]);

  const sideToMoveLabel = isWhiteTurn ? "White" : "Black";

  const pushHistory = (snap) => {
    setHistory((h) => [...h, snap]);
  };

  const snapshot = () => ({
    pieces,
    isWhiteTurn,
    enPassantTarget,
    movesSAN,
    // selection UI can be safely cleared on undo restore, but we’ll restore it too
    selectedSquare,
    legalMoves,
    // promotion states should never be active when pushing (we freeze instead),
    // but include for completeness
    promotionPending,
    frozenForPromotion,
  });

  const restoreSnapshot = (snap) => {
    setPieces(snap.pieces);
    setIsWhiteTurn(snap.isWhiteTurn);
    setEnPassantTarget(snap.enPassantTarget);
    setMovesSAN(snap.movesSAN);
    setSelectedSquare(snap.selectedSquare);
    setLegalMoves(snap.legalMoves);
    setPromotionPending(snap.promotionPending);
    setFrozenForPromotion(snap.frozenForPromotion);
  };

  const handleUndo = () => {
    if (history.length === 0 || frozenForPromotion || isThinking) return;
    const prev = history[history.length - 1];
    setHistory((h) => h.slice(0, -1));
    restoreSnapshot(prev);
  };

  const handleReset = () => {
    if (frozenForPromotion || isThinking) return;
    const fresh = makeInitialState();
    setHistory([]);
    setSelectedSquare(fresh.selectedSquare);
    setIsWhiteTurn(fresh.isWhiteTurn);
    setLegalMoves(fresh.legalMoves);
    setPromotionPending(fresh.promotionPending);
    setFrozenForPromotion(fresh.frozenForPromotion);
    setEnPassantTarget(fresh.enPassantTarget);
    setMovesSAN(fresh.movesSAN);
    setPieces(fresh.pieces);
  };

  // --- SAN helpers (kept local for AI too) ---
  const willBeEnPassant = (from, to) => {
    const mover = getPiece(pieces, from.x, from.y);
    if (!mover || mover.type !== "pawn" || !enPassantTarget) return false;
    const dx = to.x - from.x,
      dy = to.y - from.y;
    const destEmpty = !getPiece(pieces, to.x, to.y);
    const dir = mover.color === "white" ? -1 : 1;
    return (
      destEmpty &&
      dx === dir &&
      Math.abs(dy) === 1 &&
      enPassantTarget.x === to.x &&
      enPassantTarget.y === to.y
    );
  };

  function otherSameTypeCanAlsoGo(prevPieces, color, type, from, to) {
    for (let x = 0; x < 8; x++) {
      for (let y = 0; y < 8; y++) {
        if (x === from.x && y === from.y) continue;
        const p = getPiece(prevPieces, x, y);
        if (!p || p.color !== color || p.type !== type) continue;
        if (isLegalMove(prevPieces, { x, y }, to, color === "white", enPassantTarget)) {
          return { x, y };
        }
      }
    }
    return null;
  }

  function isCastle(from, to, mover) {
    return mover.type === "king" && from.x === to.x && Math.abs(to.y - from.y) === 2;
  }

  function genSAN(prevPieces, afterPieces, from, to, moverBefore, promotionType, nextEP) {
    const moving = moverBefore;
    const color = moving.color;
    const opp = enemy(color);
    const destBefore = getPiece(prevPieces, to.x, to.y);
    const enPassantCap = moving.type === "pawn" && !destBefore && willBeEnPassant(from, to);
    const wasCapture = !!destBefore || enPassantCap;

    if (isCastle(from, to, moving)) {
      let san = to.y > from.y ? "O-O" : "O-O-O";
      const inCheck = isKingInCheck(afterPieces, opp);
      const anyMoves = hasAnyLegalMove(afterPieces, opp, nextEP);
      if (inCheck) san += anyMoves ? "+" : "#";
      return san;
    }

    let san = "";
    if (moving.type === "pawn") {
      if (wasCapture) san += fileOf(from.y) + "x";
      san += sq(to.x, to.y);
      if (promotionType) san += "=" + pieceLetter[promotionType];
    } else {
      san += pieceLetter[moving.type];
      const conflict = otherSameTypeCanAlsoGo(prevPieces, color, moving.type, from, to);
      if (conflict) {
        const needFile = conflict.y !== from.y;
        const needRank = conflict.x !== from.x;
        if (needFile) san += fileOf(from.y);
        if (!needFile && needRank) san += rankOf(from.x);
        if (needFile && needRank) san += rankOf(from.x);
      }
      if (wasCapture) san += "x";
      san += sq(to.x, to.y);
    }

    const inCheck = isKingInCheck(afterPieces, opp);
    const anyMoves = hasAnyLegalMove(afterPieces, opp, nextEP);
    if (inCheck) san += anyMoves ? "+" : "#";
    return san;
  }

  // --- UI helpers ---
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

  const legalForSquare = (x, y) => legalMoves.find((m) => m.x === x && m.y === y);
  const clearSelection = () => {
    setSelectedSquare(null);
    setLegalMoves([]);
  };
  const playSound = (audioFile) => {
    new Audio(audioFile).play();
  };

  const evaluatePostMoveState = (newPieces, nextTurnIsWhite, nextEP) => {
    const side = nextTurnIsWhite ? "white" : "black";
    const inCheck = isKingInCheck(newPieces, side);
    const anyMoves = hasAnyLegalMove(newPieces, side, nextEP);
    if (inCheck) playSound(check);
    if (!anyMoves && inCheck) {
      alert(`${side} is checkmated!`);
    } else if (!anyMoves && !inCheck) {
      alert("Stalemate!");
    }
  };

  // --- Promotion (human only; AI auto-queens) ---
  const maybeStartPromotion = (afterPieces, from, to, nextTurnIsWhite, prevPieces, prevEnPassant) => {
    const moved = getPiece(afterPieces, to.x, to.y);
    if (!moved || moved.type !== "pawn") return false;
    const promoteRow = moved.color === "white" ? 0 : 7;
    if (to.x !== promoteRow) return false;

    setPromotionPending({
      from,
      to,
      color: moved.color,
      nextTurnIsWhite,
      prevPieces, // for SAN disambiguation
      prevEnPassant,
    });
    setFrozenForPromotion(true);
    return true;
  };

  const finishPromotion = (choiceType) => {
    if (!promotionPending) return;

    // take snapshot for undo BEFORE mutating
    pushHistory(snapshot());

    const { from, to, color, nextTurnIsWhite, prevPieces } = promotionPending;

    const k = `${to.x}-${to.y}`;
    const updated = { ...pieces, [k]: { color, type: choiceType, hasMoved: true } };

    const moverBefore = getPiece(prevPieces, from.x, from.y);
    const san = genSAN(prevPieces, updated, from, to, moverBefore, choiceType, /*nextEP*/ null);

    setMovesSAN((arr) => [...arr, san]);
    setPieces(updated);
    setPromotionPending(null);
    setFrozenForPromotion(false);
    setIsWhiteTurn(nextTurnIsWhite);
    clearSelection();
    evaluatePostMoveState(updated, nextTurnIsWhite, /*nextEP*/ null);
  };

  // --- Human input handlers ---
  const handleSquareClick = (x, y) => {
    if (frozenForPromotion || isThinking) return;
    const piece = getPiece(pieces, x, y);

    // Block clicks if it's an AI side
    const humanTurn =
      (isWhiteTurn && whiteMode === "human") || (!isWhiteTurn && blackMode === "human");
    if (!humanTurn) return;

    if (selectedSquare) {
      const from = { x: selectedSquare.x, y: selectedSquare.y };
      const to = { x, y };

      if (isLegalMove(pieces, from, to, isWhiteTurn, enPassantTarget)) {
        // snapshot for undo BEFORE move
        pushHistory(snapshot());

        const prev = pieces;
        const moverBefore = getPiece(prev, from.x, from.y);
        const { pieces: basePieces, nextEnPassant } = makeMove(prev, from, to, null, enPassantTarget);

        playSound(move_sound);
        clearSelection();
        setPieces(basePieces);

        const nextTurnIsWhite = !isWhiteTurn;
        setEnPassantTarget(nextEnPassant);

        if (maybeStartPromotion(basePieces, from, to, nextTurnIsWhite, prev, enPassantTarget)) return;

        const san = genSAN(prev, basePieces, from, to, moverBefore, null, nextEnPassant);
        setMovesSAN((arr) => [...arr, san]);

        setIsWhiteTurn(nextTurnIsWhite);
        evaluatePostMoveState(basePieces, nextTurnIsWhite, nextEnPassant);
        return;
      }

      if (selectedSquare.x === x && selectedSquare.y === y) {
        clearSelection();
        return;
      }

      if (piece && ((isWhiteTurn && piece.color === "white") || (!isWhiteTurn && piece.color === "black"))) {
        setSelectedSquare({ x, y });
        setLegalMoves(computeLegalMoves(x, y));
      } else {
        playSound(check);
        clearSelection();
      }
      return;
    }

    if (piece && ((isWhiteTurn && piece.color === "white") || (!isWhiteTurn && piece.color === "black"))) {
      setSelectedSquare({ x, y });
      setLegalMoves(computeLegalMoves(x, y));
    }
  };

  const handleOnDrag = (e, x, y) => {
    if (frozenForPromotion || isThinking) {
      e.preventDefault();
      return;
    }
    const humanTurn =
      (isWhiteTurn && whiteMode === "human") || (!isWhiteTurn && blackMode === "human");
    if (!humanTurn) {
      e.preventDefault();
      return;
    }
    const piece = getPiece(pieces, x, y);
    if (!piece) {
      e.preventDefault();
      return;
    }
    if ((isWhiteTurn && piece.color !== "white") || (!isWhiteTurn && piece.color !== "black")) {
      e.preventDefault();
      return;
    }
    setSelectedSquare({ x, y });
    setLegalMoves(computeLegalMoves(x, y));
    e.dataTransfer.setData("position", `${x}-${y}`);
  };

  const handleDragOver = (e) => e.preventDefault();

  const handleOnDrop = (e, x, y) => {
    if (frozenForPromotion || isThinking) return;
    const humanTurn =
      (isWhiteTurn && whiteMode === "human") || (!isWhiteTurn && blackMode === "human");
    if (!humanTurn) return;

    const data = e.dataTransfer.getData("position");
    if (!data) return;
    const [oldX, oldY] = data.split("-").map(Number);
    const from = { x: oldX, y: oldY };
    const to = { x, y };

    if (!isLegalMove(pieces, from, to, isWhiteTurn, enPassantTarget)) {
      playSound(check);
      return;
    }

    // snapshot for undo BEFORE move
    pushHistory(snapshot());

    const prev = pieces;
    const moverBefore = getPiece(prev, from.x, from.y);
    const { pieces: basePieces, nextEnPassant } = makeMove(prev, from, to, null, enPassantTarget);

    playSound(move_sound);
    clearSelection();
    setPieces(basePieces);
    setEnPassantTarget(nextEnPassant);

    const nextTurnIsWhite = !isWhiteTurn;

    if (maybeStartPromotion(basePieces, from, to, nextTurnIsWhite, prev, enPassantTarget)) return;

    const san = genSAN(prev, basePieces, from, to, moverBefore, null, nextEnPassant);
    setMovesSAN((arr) => [...arr, san]);

    setIsWhiteTurn(nextTurnIsWhite);
    evaluatePostMoveState(basePieces, nextTurnIsWhite, nextEnPassant);
  };

  // --- AI LOOP ---
  async function thinkAndMoveAI() {
    const color = isWhiteTurn ? "white" : "black";
    const mode = isWhiteTurn ? whiteMode : blackMode;

    // Stop if game is over
    const term = !hasAnyLegalMove(pieces, color, enPassantTarget);
    if (term) return;

    setIsThinking(true);

    let aiMove = null;
    if (mode === "random") {
      aiMove = await pickRandomMove(pieces, color, enPassantTarget);
    } else if (mode === "mcts") {
      aiMove = await pickMCTSMove(pieces, color, enPassantTarget, {
        timeMs: 1200,
        maxIterations: 3000,
        rolloutDepth: 40,
      });
    } else if (mode === "nn") {
      aiMove = await pickNNMove(pieces, color, enPassantTarget);
    }

    setIsThinking(false);
    if (!aiMove) return;

    // snapshot for undo BEFORE AI move
    pushHistory(snapshot());

    const prev = pieces;
    const moverBefore = getPiece(prev, aiMove.from.x, aiMove.from.y);

    // Ensure promotion default
    if (aiMove.needsPromotion && !aiMove.promotionType) aiMove.promotionType = "queen";

    const { pieces: basePieces, nextEnPassant } = makeMove(
      prev,
      aiMove.from,
      aiMove.to,
      aiMove.promotionType,
      enPassantTarget
    );

    playSound(move_sound);
    setPieces(basePieces);
    setEnPassantTarget(nextEnPassant);
    clearSelection();

    const san = genSAN(
      prev,
      basePieces,
      aiMove.from,
      aiMove.to,
      moverBefore,
      aiMove.promotionType ?? null,
      nextEnPassant
    );
    setMovesSAN((arr) => [...arr, san]);

    const nextTurnIsWhite = !isWhiteTurn;
    setIsWhiteTurn(nextTurnIsWhite);
    evaluatePostMoveState(basePieces, nextTurnIsWhite, nextEnPassant);
  }

  // Trigger AI after any state change that gives turn to an AI side
  useEffect(() => {
    const mode = isWhiteTurn ? whiteMode : blackMode;
    if (mode === "human") return;
    const id = setTimeout(() => {
      thinkAndMoveAI();
    }, 50);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isWhiteTurn, whiteMode, blackMode, pieces, enPassantTarget]);

  // --- Render ---
  const pgnText = useMemo(() => {
    const parts = [];
    for (let i = 0; i < movesSAN.length; i += 2) {
      const moveNo = i / 2 + 1;
      const w = movesSAN[i] ?? "";
      const b = movesSAN[i + 1] ?? "";
      parts.push(b ? `${moveNo}. ${w} ${b}` : `${moveNo}. ${w}`);
    }
    return parts.join(" ");
  }, [movesSAN]);

  const modes = [
    { key: "human", label: "Human" },
    { key: "random", label: "Random" },
    { key: "mcts", label: "MCTS" },
    { key: "nn", label: "NN" },
  ];

  return (
    <div className="chess-layout">
      <div className="chess-sidepanel">
        <div className="turn-indicator">
          <b>Turn:</b> {sideToMoveLabel} {isThinking ? " (AI is thinking...)" : ""}
        </div>

        <div className="mode-row">
          <div>
            <b>White:</b>
          </div>
          <div className="segmented">
            {modes.map((m) => (
              <button
                key={m.key}
                className={`seg-btn ${whiteMode === m.key ? "is-active" : ""}`}
                onClick={() => setWhiteMode(m.key)}
                disabled={isThinking || frozenForPromotion}
                type="button"
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <div className="mode-row">
          <div>
            <b>Black:</b>
          </div>
          <div className="segmented">
            {modes.map((m) => (
              <button
                key={m.key}
                className={`seg-btn ${blackMode === m.key ? "is-active" : ""}`}
                onClick={() => setBlackMode(m.key)}
                disabled={isThinking || frozenForPromotion}
                type="button"
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <div className="pgn-title">PGN</div>
        <div className="pgn-box">{pgnText || "—"}</div>
      </div>

      <div className="board-area">
        <div className="board-topbar">
          <div className="board-actions">
            <button className="ui-btn" onClick={handleReset} disabled={isThinking || frozenForPromotion} type="button">
              Reset
            </button>
            <button className="ui-btn" onClick={handleUndo} disabled={history.length === 0 || isThinking || frozenForPromotion} type="button">
              Undo
            </button>
          </div>

          <div className="board-actions">
            <button
              className="ui-btn"
              onClick={() => setIsFlipped((v) => !v)}
              disabled={isThinking || frozenForPromotion}
              type="button"
            >
              Flip Board
            </button>
          </div>
        </div>

        {/* Promotion modal */}
        {promotionPending && (
          <div className="promotion-modal">
            <div className="promotion-card">
              <div className="promotion-title">Choose promotion</div>
              <div className="promotion-choices">
                <button onClick={() => finishPromotion("queen")} type="button">
                  Queen
                </button>
                <button onClick={() => finishPromotion("rook")} type="button">
                  Rook
                </button>
                <button onClick={() => finishPromotion("bishop")} type="button">
                  Bishop
                </button>
                <button onClick={() => finishPromotion("knight")} type="button">
                  Knight
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="chessboard-wrap">
          <div className={`chessboard ${isFlipped ? "is-flipped" : ""}`}>
            {Array.from({ length: 8 }, (_, y) => (
              <div key={y} className="chessboard-row">
                {Array.from({ length: 8 }, (_, x) => {

                  // ✅ map VIEW coords -> BOARD coords (mirror files)
                  const bx = x;
                  const by = 7 - y;

                  const piece = getPiece(pieces, bx, by);
                  const selected =
                    selectedSquare && selectedSquare.x === bx && selectedSquare.y === by;

                  const lm = legalForSquare(bx, by);
                  const isCapture = lm?.capture;

                  // screen-space indices in loops:
                  // x = col (0..7 left->right)
                  // y = row (0..7 top->bottom)

                  // Labels based on BOARD coords (so rotation doesn’t swap edges visually)
                  const isBottomEdge = (bx === 7); // rank 1 side
                  const isLeftEdge = (by === 0);   // file a side

                  const fileLabel = String.fromCharCode(97 + by); // a..h
                  const rankLabel = String(8 - bx);               // 8..1



                  return (
                    <div
                      key={`${bx}-${by}`}
                      className={[
                        "chessboard-square",
                        // keep colors based on VIEW coords for user familiarity
                        (x + y) % 2 === 1 ? "chessboard-square--black" : "chessboard-square--white",
                        selected ? "chessboard-square--selected" : "",
                        lm ? (isCapture ? "chessboard-square--legal-capture" : "chessboard-square--legal-move") : "",
                        frozenForPromotion || isThinking ? "chessboard-square--disabled" : "",
                      ].join(" ")}
                      // ✅ call handlers with BOARD coords
                      onClick={() => handleSquareClick(bx, by)}
                      onDrop={(e) => handleOnDrop(e, bx, by)}
                      onDragOver={handleDragOver}
                    >
                      {isBottomEdge && <span className="sq-label file">{fileLabel}</span>}
                      {isLeftEdge && <span className="sq-label rank">{rankLabel}</span>}



                      <div
                        className="chesspiece-wrapper"
                        draggable={
                          !!piece &&
                          !(frozenForPromotion || isThinking) &&
                          ((isWhiteTurn && piece.color === "white") ||
                            (!isWhiteTurn && piece.color === "black")) &&
                          ((isWhiteTurn && whiteMode === "human") ||
                            (!isWhiteTurn && blackMode === "human"))
                        }
                        // ✅ drag-start must also use BOARD coords so dataTransfer matches
                        onDragStart={(e) => handleOnDrag(e, bx, by)}
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
      </div>
    </div>
  );
}

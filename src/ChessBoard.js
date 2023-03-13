import React, { useState } from 'react';
import './index.css';
import ChessPiece from './ChessPiece';


const ChessBoard = () => {
    // Define the initial state for the selected square
    const [selectedSquare, setSelectedSquare] = useState(null);

    const [pieces, setPieces] = useState({
        '0-0': { color: 'black', type: 'rook' },
        '0-1': { color: 'black', type: 'knight' },
        '0-2': { color: 'black', type: 'bishop' },
        '0-3': { color: 'black', type: 'queen' },
        '0-4': { color: 'black', type: 'king' },
        '0-5': { color: 'black', type: 'bishop' },
        '0-6': { color: 'black', type: 'knight' },
        '0-7': { color: 'black', type: 'rook' },
        '1-0': { color: 'black', type: 'pawn' },
        '1-1': { color: 'black', type: 'pawn' },
        '1-2': { color: 'black', type: 'pawn' },
        '1-3': { color: 'black', type: 'pawn' },
        '1-4': { color: 'black', type: 'pawn' },
        '1-5': { color: 'black', type: 'pawn' },
        '1-6': { color: 'black', type: 'pawn' },
        '1-7': { color: 'black', type: 'pawn' },
        '6-0': { color: 'white', type: 'pawn' },
        '6-1': { color: 'white', type: 'pawn' },
        '6-2': { color: 'white', type: 'pawn' },
        '6-3': { color: 'white', type: 'pawn' },
        '6-4': { color: 'white', type: 'pawn' },
        '6-5': { color: 'white', type: 'pawn' },
        '6-6': { color: 'white', type: 'pawn' },
        '6-7': { color: 'white', type: 'pawn' },
        '7-0': { color: 'white', type: 'rook' },
        '7-1': { color: 'white', type: 'knight' },
        '7-2': { color: 'white', type: 'bishop' },
        '7-3': { color: 'white', type: 'queen' },
        '7-4': { color: 'white', type: 'king' },
        '7-5': { color: 'white', type: 'bishop' },
        '7-6': { color: 'white', type: 'knight' },
        '7-7': { color: 'white', type: 'rook' },
    });

    // Define a helper function to determine if a square is black or white
    const isBlackSquare = (x, y) => {
        return (x + y) % 2 === 1;
    };

    // Define a helper function to handle clicks on a square
    const handleSquareClick = (x, y) => {
        console.log(`Clicked square: ${x}, ${y}`);
        // If the square is already selected, deselect it
        if (selectedSquare && selectedSquare.x === x && selectedSquare.y === y) {
            setSelectedSquare(null);
        } else {
            // Otherwise, select the square
            setSelectedSquare({ x, y });
        }
    };

    // Render the chessboard layout
    return (
        <div className="chessboard">
            {Array.from({ length: 8 }, (_, y) => (
                <div key={y} className="chessboard-row">
                    {Array.from({ length: 8 }, (_, x) => (
                        <div
                            key={`${x}-${y}`}
                            className={`chessboard-square ${isBlackSquare(x, y) ? 'chessboard-square--black' : 'chessboard-square--white'
                                } ${selectedSquare && selectedSquare.x === x && selectedSquare.y === y
                                    ? 'chessboard-square--selected'
                                    : ''
                                }
                                ${selectedSquare && selectedSquare.x === x && selectedSquare.y === y && !pieces[`${x}-${y}`] ? 'chessboard-square--empty' : ''}
                                    `}
                            onClick={() => handleSquareClick(x, y)}

                        >
                            {/* Render the piece here */}
                            <ChessPiece piece={pieces[`${x}-${y}`]} />
                        </div>
                    ))}
                </div>
            ))
            }
        </div >
    );
};

export default ChessBoard;

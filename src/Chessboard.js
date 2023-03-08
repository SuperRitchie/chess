import React, { useState } from 'react';
import './index.css';

const Chessboard = () => {
    // Define the initial state for the selected square
    const [selectedSquare, setSelectedSquare] = useState(null);

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
                                }`}
                            onClick={() => handleSquareClick(x, y)}
                        >
                            {/* Render the piece here */}
                            {/* If the square is selected, render the piece */}

                        </div>
                    ))}
                </div>
            ))
            }
        </div >
    );
};

export default Chessboard;

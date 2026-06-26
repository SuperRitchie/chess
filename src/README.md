# source layout

```text
src/
  ai/          chess computer players
  assets/      piece SVGs and sound effects
  components/  reusable React components and component styles
  rules/       chess move generation and legality rules
```

`ChessBoard.js` still imports through the top-level compatibility wrappers so this refactor stays low-risk. New code should import directly from the folder that owns the implementation.

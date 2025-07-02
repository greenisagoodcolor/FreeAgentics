// Mock for nanoid to handle ESM incompatibility in Jest
module.exports = {
  nanoid: () => {
    // Generate a random ID for testing
    return (
      Math.random().toString(36).substring(2, 15) +
      Math.random().toString(36).substring(2, 15)
    );
  },
  customAlphabet: (alphabet, size) => {
    return () => {
      let result = "";
      for (let i = 0; i < size; i++) {
        result += alphabet.charAt(Math.floor(Math.random() * alphabet.length));
      }
      return result;
    };
  },
};

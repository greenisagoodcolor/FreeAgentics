// Test file to prevent "no tests found" error for fileMock.js
describe("fileMock", () => {
  it("should export a string", () => {
    const fileMock = require("./fileMock.js");
    expect(typeof fileMock).toBe("string");
    expect(fileMock).toBe("test-file-stub");
  });
});

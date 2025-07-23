// Test for fileMock to prevent Jest "no tests found" warnings
describe("fileMock", () => {
  it("should export test file stub", () => {
    const fileMock = require("./fileMock.js");
    expect(fileMock).toBe("test-file-stub");
  });
});

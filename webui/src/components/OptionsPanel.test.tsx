import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { DEFAULT_OPTIONS, normalizeOptions } from "../lib/options";
import { OptionsPanel } from "./OptionsPanel";

const renderPanel = (
  optionsPatch: Partial<typeof DEFAULT_OPTIONS>,
  onChange = vi.fn(),
) => {
  render(
    <OptionsPanel
      options={normalizeOptions({ ...DEFAULT_OPTIONS, ...optionsPatch })}
      onChange={onChange}
      modelSuggestions={[]}
    />,
  );
  return onChange;
};

describe("OptionsPanel", () => {
  it("disables diarization and speaker embeddings until alignment is on", () => {
    renderPanel({ align: false });
    expect(screen.getByLabelText(/Diarize speakers/)).toBeDisabled();
    expect(screen.getByLabelText(/Speaker embeddings/)).toBeDisabled();
  });

  it("enables diarization with alignment, and embeddings once diarizing", () => {
    renderPanel({ align: true, diarize: true });
    expect(screen.getByLabelText(/Diarize speakers/)).toBeEnabled();
    expect(screen.getByLabelText(/Speaker embeddings/)).toBeEnabled();
  });

  it("cascades the invariants through onChange when alignment is turned off", async () => {
    const onChange = renderPanel({ align: true, diarize: true });
    await userEvent.click(screen.getByLabelText(/Align timestamps/));
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ align: false, diarize: false }),
    );
  });
});

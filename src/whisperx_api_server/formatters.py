from whisperx.utils import WriteSRT, WriteVTT, WriteAudacity

class ListWriter:
    """Helper class to store written lines in memory."""
    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(text)

    def get_output(self):
        return ''.join(self.lines)

    def flush(self):
        pass

def update_options(kwargs, defaults):
    """
    Helper function to update default options with values from kwargs.
    
    :param kwargs: Keyword arguments from the function call.
    :param defaults: Dictionary of default values.
    :return: Updated options dictionary.
    """
    options = defaults.copy()
    options.update({key: kwargs.get(key, value) for key, value in defaults.items()})
    return options

def handle_whisperx_format(transcript, writer_class, options):
    """
    Helper function to handle "srt", "vtt" and "aud" formats using whisperx writers.
    
    :param transcript: The transcript dictionary.
    :param writer_class: The writer class (WriteSRT, WriteVTT or WriteAudacity).
    :param options: Options for the writer.
    :return: Formatted output as a string.
    """
    writer = writer_class(output_dir=None)
    output = ListWriter()

    transcript["segments"]["language"] = transcript["language"]
    
    writer.write_result(transcript["segments"], output, options)

    return output.get_output()

def format_transcription(transcript, format, **kwargs):
    """
    Format a transcript into a given format.
    
    :param transcript: The transcript to format, a dictionary with a "segments" key that contains a list of segments with start and end times and text.
    :param format: The format to generate the transcript in. Supported formats are "json", "text", "srt", "vtt" and "aud".
    :param kwargs: Additional keyword arguments to pass to the formatter.
    :return: The formatted transcript, a string if format is "text", "srt", "vtt" or "aud", or a JSON-serializable dictionary if format is "json" or "verbose_json".
    """
    # Default options, used for formats imported from whisperx.utils
    defaults = {
        "max_line_width": 1000,
        "max_line_count": None,
        "highlight_words": kwargs.get("highlight_words", True)
    }
    options = update_options(kwargs, defaults)

    if format == "json":
        return {"text": transcript.get("text", "")}
    elif format == "verbose_json":
        return transcript
    elif format == "vtt_json":
        transcript["vtt_text"] = handle_whisperx_format(transcript, WriteVTT, options)
        return transcript
    elif format == "text":
        return transcript.get("text", "")
    elif format == "srt":
        return handle_whisperx_format(transcript, WriteSRT, options)
    elif format == "vtt":
        return handle_whisperx_format(transcript, WriteVTT, options)
    elif format == "aud":
        return handle_whisperx_format(transcript, WriteAudacity, options)
    else:
        raise ValueError(f"Unsupported format: {format}")

import unittest

from core.constitutional_brain import AudioProcessor, VisionProcessor


class VisionBackend:
    def detect_objects(self, image_data):
        return [f"detected:{image_data}"]

    def analyze_scene(self, image_data):
        return {"scene_category": "backend", "input": image_data}


class AudioBackend:
    def transcribe_speech(self, audio_data):
        return f"transcribed:{audio_data}"


class MultimodalBackendContractTests(unittest.TestCase):
    def test_vision_backend_is_used_when_available(self):
        processor = VisionProcessor(backend=VisionBackend())
        self.assertEqual(processor.detect_objects("frame"), ["detected:frame"])
        self.assertEqual(
            processor.analyze_scene("frame"),
            {"scene_category": "backend", "input": "frame"},
        )

    def test_vision_fallback_is_explicit_when_backend_is_absent(self):
        processor = VisionProcessor()
        self.assertEqual(processor.detect_objects(None), ["person", "chair", "table", "laptop"])
        self.assertEqual(processor.analyze_scene(None)["scene_category"], "office")

    def test_audio_backend_is_used_when_available(self):
        processor = AudioProcessor(backend=AudioBackend())
        self.assertEqual(processor.transcribe_speech("clip"), "transcribed:clip")

    def test_audio_fallback_is_explicit_when_backend_is_absent(self):
        processor = AudioProcessor()
        self.assertEqual(
            processor.transcribe_speech(None),
            "This is a simulated speech transcription",
        )


if __name__ == "__main__":
    unittest.main()

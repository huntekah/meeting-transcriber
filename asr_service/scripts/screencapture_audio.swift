#!/usr/bin/env swift

import AVFoundation
import ScreenCaptureKit
import Foundation

/// Simple ScreenCaptureKit audio capturer that outputs raw PCM to stdout
/// Usage: swift screencapture_audio.swift [sample_rate] [duration]

@available(macOS 13.0, *)
class ScreenAudioCapturer: NSObject, SCStreamDelegate, SCStreamOutput {
    private var stream: SCStream?
    private let sampleRate: Int
    private var isCapturing = false

    init(sampleRate: Int = 16000) {
        self.sampleRate = sampleRate
        super.init()
    }

    func startCapture() async throws {
        // Get available content (need to request permission)
        let availableContent = try await SCShareableContent.excludingDesktopWindows(
            false,
            onScreenWindowsOnly: true
        )

        // Create filter to capture system audio
        let filter = SCContentFilter(
            display: availableContent.displays[0],
            excludingApplications: [],
            exceptingWindows: []
        )

        // Configure stream for audio capture
        let streamConfig = SCStreamConfiguration()
        streamConfig.capturesAudio = true
        streamConfig.sampleRate = sampleRate
        streamConfig.channelCount = 1  // Mono

        // Disable video capture
        streamConfig.width = 1
        streamConfig.height = 1
        streamConfig.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        // Create and start stream
        stream = SCStream(filter: filter, configuration: streamConfig, delegate: self)

        try stream?.addStreamOutput(self, type: .audio, sampleHandlerQueue: .main)
        try await stream?.startCapture()

        isCapturing = true
        fputs("INFO: ScreenCaptureKit audio capture started\n", stderr)
    }

    func stopCapture() async throws {
        try await stream?.stopCapture()
        isCapturing = false
        fputs("INFO: ScreenCaptureKit audio capture stopped\n", stderr)
    }

    // SCStreamOutput - handle audio samples
    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of outputType: SCStreamOutputType) {
        guard outputType == .audio else { return }

        // Get audio buffer
        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else {
            return
        }

        // Get audio format
        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
              let streamBasicDescription = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription) else {
            return
        }

        // Read audio data
        var length = 0
        var dataPointer: UnsafeMutablePointer<Int8>?

        let status = CMBlockBufferGetDataPointer(
            blockBuffer,
            atOffset: 0,
            lengthAtOffsetOut: nil,
            totalLengthOut: &length,
            dataPointerOut: &dataPointer
        )

        guard status == kCMBlockBufferNoErr,
              let data = dataPointer else {
            return
        }

        // Convert to Data and write to stdout as raw PCM
        let audioData = Data(bytes: data, count: length)

        // Write raw PCM to stdout
        FileHandle.standardOutput.write(audioData)
    }

    // SCStreamDelegate - handle errors
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("ERROR: Stream stopped with error: \(error)\n", stderr)
        isCapturing = false
    }
}

@available(macOS 13.0, *)
func main() async {
    let args = CommandLine.arguments
    let sampleRate = args.count > 1 ? Int(args[1]) ?? 16000 : 16000
    let duration = args.count > 2 ? Int(args[2]) ?? 10 : 10

    fputs("INFO: Starting ScreenCaptureKit audio capture\n", stderr)
    fputs("INFO: Sample rate: \(sampleRate) Hz\n", stderr)
    fputs("INFO: Duration: \(duration) seconds\n", stderr)
    fputs("INFO: Output: Raw PCM (mono, float32)\n", stderr)

    // Request permission
    fputs("INFO: Requesting screen recording permission...\n", stderr)
    let granted = await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)

    let capturer = ScreenAudioCapturer(sampleRate: sampleRate)

    do {
        try await capturer.startCapture()

        // Capture for specified duration
        try await Task.sleep(nanoseconds: UInt64(duration) * 1_000_000_000)

        try await capturer.stopCapture()
        fputs("INFO: Capture complete\n", stderr)

    } catch {
        fputs("ERROR: \(error)\n", stderr)
        exit(1)
    }
}

if #available(macOS 13.0, *) {
    Task {
        await main()
    }

    // Keep running
    RunLoop.main.run()
} else {
    fputs("ERROR: ScreenCaptureKit requires macOS 13.0 or later\n", stderr)
    exit(1)
}

#!/usr/bin/env swift

import Foundation
import CoreAudio

func fourCCString(_ code: UInt32) -> String {
    let bytes: [UInt8] = [
        UInt8((code >> 24) & 0xff),
        UInt8((code >> 16) & 0xff),
        UInt8((code >> 8) & 0xff),
        UInt8(code & 0xff),
    ]
    return String(bytes: bytes, encoding: .macOSRoman) ?? String(format: "0x%08x", code)
}

func checkStatus(_ status: OSStatus, _ message: String) throws {
    guard status == noErr else {
        throw NSError(domain: "CoreAudioTap", code: Int(status), userInfo: [
            NSLocalizedDescriptionKey: "\(message) failed (OSStatus=\(status))"
        ])
    }
}

func getDefaultSystemOutputDevice() throws -> AudioObjectID {
    var deviceID = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultSystemOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMaster
    )
    try checkStatus(
        AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0,
            nil,
            &size,
            &deviceID
        ),
        "AudioObjectGetPropertyData(default system output device)"
    )
    return deviceID
}

func getDeviceUID(_ deviceID: AudioObjectID) throws -> String {
    var uid: CFString? = nil
    var size = UInt32(MemoryLayout<CFString?>.size)
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyDeviceUID,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMaster
    )
    try checkStatus(
        AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &uid),
        "AudioObjectGetPropertyData(device UID)"
    )
    return (uid as String?) ?? ""
}

func getTapUID(_ tapID: AudioObjectID) throws -> String {
    var uid: CFString? = nil
    var size = UInt32(MemoryLayout<CFString?>.size)
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioTapPropertyUID,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMaster
    )
    try checkStatus(
        AudioObjectGetPropertyData(tapID, &address, 0, nil, &size, &uid),
        "AudioObjectGetPropertyData(tap UID)"
    )
    return (uid as String?) ?? ""
}

func getTapFormat(_ tapID: AudioObjectID) throws -> AudioStreamBasicDescription {
    var asbd = AudioStreamBasicDescription()
    var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioTapPropertyFormat,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMaster
    )
    try checkStatus(
        AudioObjectGetPropertyData(tapID, &address, 0, nil, &size, &asbd),
        "AudioObjectGetPropertyData(tap format)"
    )
    return asbd
}

@available(macOS 14.2, *)
func runCapture(sampleRate: Double, durationSeconds: Int) throws {
    let deviceID = try getDefaultSystemOutputDevice()
    let deviceUID = try getDeviceUID(deviceID)

    let tapDescription = CATapDescription(monoGlobalTapButExcludeProcesses: [])
    tapDescription.name = "MeetingScribe Tap"
    tapDescription.deviceUID = deviceUID
    tapDescription.muteBehavior = CATapUnmuted

    var tapID = AudioObjectID(kAudioObjectUnknown)
    try checkStatus(AudioHardwareCreateProcessTap(tapDescription, &tapID), "AudioHardwareCreateProcessTap")
    defer {
        _ = AudioHardwareDestroyProcessTap(tapID)
    }

    let tapUID = try getTapUID(tapID)
    fputs("INFO: Tap UID: \(tapUID)\n", stderr)

    let aggregateUID = "com.meeting_scribe.tap.\(UUID().uuidString)"
    let aggregateName = "MeetingScribe Tap Aggregate"
    let tapList: [[String: Any]] = [
        [kAudioSubTapUIDKey: tapUID]
    ]
    let aggregateDescription: [String: Any] = [
        kAudioAggregateDeviceUIDKey: aggregateUID,
        kAudioAggregateDeviceNameKey: aggregateName,
        kAudioAggregateDeviceIsPrivateKey: 1,
        kAudioAggregateDeviceTapListKey: tapList,
        kAudioAggregateDeviceClockDeviceKey: deviceUID,
        kAudioAggregateDeviceTapAutoStartKey: 1
    ]

    var aggregateID = AudioObjectID(kAudioObjectUnknown)
    try checkStatus(
        AudioHardwareCreateAggregateDevice(aggregateDescription as CFDictionary, &aggregateID),
        "AudioHardwareCreateAggregateDevice"
    )
    defer {
        _ = AudioHardwareDestroyAggregateDevice(aggregateID)
    }

    var desiredSampleRate = sampleRate
    var sampleRateSize = UInt32(MemoryLayout<Double>.size)
    var sampleRateAddress = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyNominalSampleRate,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMaster
    )
    let srStatus = AudioObjectSetPropertyData(
        aggregateID,
        &sampleRateAddress,
        0,
        nil,
        sampleRateSize,
        &desiredSampleRate
    )
    if srStatus != noErr {
        fputs("WARN: Failed to set sample rate to \(sampleRate) (OSStatus=\(srStatus))\n", stderr)
    }

    let tapFormat = try getTapFormat(tapID)
    let formatID = fourCCString(tapFormat.mFormatID)
    fputs(
        "INFO: Tap format: sampleRate=\(tapFormat.mSampleRate), channels=\(tapFormat.mChannelsPerFrame), " +
        "formatID=\(formatID), bitsPerChannel=\(tapFormat.mBitsPerChannel), flags=0x\(String(tapFormat.mFormatFlags, radix: 16))\n",
        stderr
    )

    var ioProcID: AudioDeviceIOProcID? = nil
    let ioBlock: AudioDeviceIOBlock = { _, inInputData, _, _, _ in
        guard let inInputData = inInputData else { return }
        let bufferList = UnsafeMutableAudioBufferListPointer(mutating: inInputData)
        for buffer in bufferList {
            if let mData = buffer.mData, buffer.mDataByteSize > 0 {
                let data = Data(bytes: mData, count: Int(buffer.mDataByteSize))
                FileHandle.standardOutput.write(data)
            }
        }
    }

    try checkStatus(
        AudioDeviceCreateIOProcIDWithBlock(&ioProcID, aggregateID, nil, ioBlock),
        "AudioDeviceCreateIOProcIDWithBlock"
    )
    guard let startedIOProcID = ioProcID else {
        throw NSError(domain: "CoreAudioTap", code: -1, userInfo: [
            NSLocalizedDescriptionKey: "Failed to create IOProc"
        ])
    }
    defer {
        _ = AudioDeviceDestroyIOProcID(aggregateID, startedIOProcID)
    }

    try checkStatus(AudioDeviceStart(aggregateID, startedIOProcID), "AudioDeviceStart")
    defer {
        _ = AudioDeviceStop(aggregateID, startedIOProcID)
    }

    fputs("INFO: CoreAudio tap capture started\n", stderr)
    Thread.sleep(forTimeInterval: TimeInterval(durationSeconds))
    fputs("INFO: CoreAudio tap capture finished\n", stderr)
}

let args = CommandLine.arguments
let sampleRate = args.count > 1 ? Double(args[1]) ?? 16000.0 : 16000.0
let duration = args.count > 2 ? Int(args[2]) ?? 10 : 10

if #available(macOS 14.2, *) {
    do {
        try runCapture(sampleRate: sampleRate, durationSeconds: duration)
    } catch {
        fputs("ERROR: \(error)\n", stderr)
        exit(1)
    }
} else {
    fputs("ERROR: CoreAudio taps require macOS 14.2 or later\n", stderr)
    exit(1)
}

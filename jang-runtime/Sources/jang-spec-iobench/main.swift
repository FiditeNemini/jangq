// jang-spec-iobench — measure Mac NVMe -> unified memory throughput
// via MTLIOCommandQueue, compared to plain pread.
//
// Fixtures: creates N files of SIZE bytes each under a tmpdir, fills them
// with a fast PRNG (not zeros — apfs/nvme may shortcut zero reads), then:
//   (1) sequential MTLIOCommandQueue reads into MTLBuffer
//   (2) random-order MTLIOCommandQueue reads
//   (3) random-order pread into a Data buffer
// and reports GB/s + per-read latency p50/p99.

import Foundation
import Metal
import MetalKit

// ----- Config -----
let NUM_FILES = 256
let FILE_BYTES = 50 * 1024 * 1024   // 50 MB per file, ~one expert's worth
let ALIGN = 4096

@inline(__always)
func nowNs() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

func xorshift(_ s: inout UInt64) -> UInt64 {
    var x = s
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    s = x
    return x
}

func fillRandom(_ buf: UnsafeMutableRawPointer, _ len: Int, seed: UInt64) {
    var s = seed | 1
    let p = buf.assumingMemoryBound(to: UInt64.self)
    let n = len / 8
    for i in 0..<n {
        p[i] = xorshift(&s)
    }
}

func percentiles(_ xs: [Double]) -> (p50: Double, p99: Double) {
    let s = xs.sorted()
    let p50 = s[s.count / 2]
    let p99 = s[min(s.count - 1, Int(Double(s.count) * 0.99))]
    return (p50, p99)
}

func makeFixtures(dir: URL) throws -> [URL] {
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    var urls: [URL] = []
    let buf = UnsafeMutableRawPointer.allocate(byteCount: FILE_BYTES, alignment: ALIGN)
    defer { buf.deallocate() }

    print("  creating \(NUM_FILES) × \(FILE_BYTES / 1024 / 1024) MB fixture files under \(dir.path) ...")
    let t0 = nowNs()
    for i in 0..<NUM_FILES {
        let url = dir.appendingPathComponent(String(format: "f-%05d.bin", i))
        fillRandom(buf, FILE_BYTES, seed: UInt64(i + 1))
        let data = Data(bytesNoCopy: buf, count: FILE_BYTES, deallocator: .none)
        try data.write(to: url)
        urls.append(url)
    }
    let elapsed = Double(nowNs() - t0) / 1e9
    let totalGB = Double(NUM_FILES * FILE_BYTES) / 1e9
    print(String(format: "    %.2f GB written in %.1fs (%.1f GB/s)", totalGB, elapsed, totalGB / elapsed))
    // Flush the page cache by running `purge` — optional, requires sudo, best-effort.
    return urls
}

func benchIOCommandQueue(device: MTLDevice, urls: [URL], random: Bool) throws -> (gbPerSec: Double, p50ms: Double, p99ms: Double) {
    guard #available(macOS 13.0, *) else {
        print("  MTLIOCommandQueue requires macOS 13+")
        exit(1)
    }
    let ioQueue: MTLIOCommandQueue
    let desc = MTLIOCommandQueueDescriptor()
    desc.type = .concurrent
    desc.priority = .normal
    ioQueue = try device.makeIOCommandQueue(descriptor: desc)

    // Buffers: one per file.
    var buffers: [MTLBuffer] = []
    buffers.reserveCapacity(urls.count)
    for _ in urls {
        guard let b = device.makeBuffer(length: FILE_BYTES, options: [.storageModeShared]) else {
            throw NSError(domain: "iobench", code: 1)
        }
        buffers.append(b)
    }

    // File handles for the IO queue.
    var handles: [MTLIOFileHandle] = []
    for url in urls {
        let h = try device.makeIOFileHandle(url: url)
        handles.append(h)
    }

    // Build an order.
    var order = Array(0..<urls.count)
    if random { order.shuffle() }

    var latencies: [Double] = []
    latencies.reserveCapacity(order.count)

    let tStart = nowNs()
    for i in order {
        let cb = ioQueue.makeCommandBuffer()
        cb.load(
            buffers[i],
            offset: 0,
            size: FILE_BYTES,
            sourceHandle: handles[i],
            sourceHandleOffset: 0
        )
        let before = nowNs()
        cb.commit()
        cb.waitUntilCompleted()
        let after = nowNs()
        latencies.append(Double(after - before) / 1e6) // ms
    }
    let elapsed = Double(nowNs() - tStart) / 1e9
    let totalGB = Double(order.count * FILE_BYTES) / 1e9
    let (p50, p99) = percentiles(latencies)
    return (totalGB / elapsed, p50, p99)
}

func benchPread(urls: [URL], random: Bool) throws -> (gbPerSec: Double, p50ms: Double, p99ms: Double) {
    var order = Array(0..<urls.count)
    if random { order.shuffle() }

    let buf = UnsafeMutableRawPointer.allocate(byteCount: FILE_BYTES, alignment: ALIGN)
    defer { buf.deallocate() }

    var latencies: [Double] = []
    latencies.reserveCapacity(order.count)

    let tStart = nowNs()
    for i in order {
        let fd = open(urls[i].path, O_RDONLY)
        if fd < 0 { throw NSError(domain: "iobench", code: 2) }
        defer { close(fd) }
        let before = nowNs()
        var off: off_t = 0
        var remaining = FILE_BYTES
        var p = buf
        while remaining > 0 {
            let n = pread(fd, p, remaining, off)
            if n <= 0 { break }
            remaining -= n
            off += off_t(n)
            p = p.advanced(by: n)
        }
        let after = nowNs()
        latencies.append(Double(after - before) / 1e6)
    }
    let elapsed = Double(nowNs() - tStart) / 1e9
    let totalGB = Double(order.count * FILE_BYTES) / 1e9
    let (p50, p99) = percentiles(latencies)
    return (totalGB / elapsed, p50, p99)
}

// ----- main -----
let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("jang-spec-iobench")
defer { try? FileManager.default.removeItem(at: tmp) }

guard let device = MTLCreateSystemDefaultDevice() else {
    print("no Metal device")
    exit(1)
}
print("  device: \(device.name)")

let urls = try makeFixtures(dir: tmp)

print("\n  (1) MTLIOCommandQueue, sequential order")
let seq = try benchIOCommandQueue(device: device, urls: urls, random: false)
print(String(format: "    throughput: %.2f GB/s   p50: %.2f ms   p99: %.2f ms", seq.gbPerSec, seq.p50ms, seq.p99ms))

print("\n  (2) MTLIOCommandQueue, random order")
let rnd = try benchIOCommandQueue(device: device, urls: urls, random: true)
print(String(format: "    throughput: %.2f GB/s   p50: %.2f ms   p99: %.2f ms", rnd.gbPerSec, rnd.p50ms, rnd.p99ms))

print("\n  (3) pread, random order")
let pr = try benchPread(urls: urls, random: true)
print(String(format: "    throughput: %.2f GB/s   p50: %.2f ms   p99: %.2f ms", pr.gbPerSec, pr.p50ms, pr.p99ms))

print("\n  verdict:")
if rnd.gbPerSec >= 3.0 && rnd.p99ms <= 5.0 {
    print("    GO — random-access streaming meets design thresholds")
} else {
    print(String(format: "    REVISIT — want >= 3 GB/s random and <= 5 ms p99, got %.2f GB/s / %.2f ms",
                 rnd.gbPerSec, rnd.p99ms))
}

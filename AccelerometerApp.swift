import SwiftUI
import CoreMotion
import Charts
import WatchConnectivity

private let baseURL = "https://web-production-8cb5b.up.railway.app"
private let sensorRecorderDuration: TimeInterval = 12 * 60 * 60
private let recordedSessionStartKey = "recordedSessionStart"
private let recordedSessionEndKey = "recordedSessionEnd"

// MARK: - Data Models

struct SessionSummary: Codable, Identifiable {
    let id: String
    let startedAt: String
    let durationS: Double
    let sampleCount: Int
    let analysis: AnalysisResult?

    enum CodingKeys: String, CodingKey {
        case id = "session_id"
        case startedAt = "started_at"
        case durationS = "duration_s"
        case sampleCount = "sample_count"
        case analysis
    }
}

struct AnalysisResult: Codable {
    let tau: Double?
    let powerLawRange: Double?
    let goodnessOfFit: Double?
    let isScaleFree: Bool?
    let nEvents: Int?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case tau
        case powerLawRange = "power_law_range"
        case goodnessOfFit = "goodness_of_fit"
        case isScaleFree = "is_scale_free"
        case nEvents = "n_events"
        case error
    }
}

struct SampleEntry: Codable {
    let t: Double
    let x: Double
    let y: Double
    let z: Double
    let magnitude: Double
}

struct SessionUploadPayload: Encodable {
    let userID: String
    let userName: String
    let startedAt: String
    let samples: [SampleEntry]

    enum CodingKeys: String, CodingKey {
        case userID = "user_id"
        case userName = "user_name"
        case startedAt = "started_at"
        case samples
    }
}

struct WatchSessionTransferPayload: Decodable {
    let startedAt: String
    let samples: [SampleEntry]

    enum CodingKeys: String, CodingKey {
        case startedAt = "started_at"
        case samples
    }
}

// MARK: - AccelerometerManager

@Observable
class AccelerometerManager: NSObject, WCSessionDelegate {
    let userID: String
    var userName: String

    var isRecording = false
    var elapsedTime: Double = 0
    var currentX: Double = 0
    var currentY: Double = 0
    var currentZ: Double = 0
    var currentMagnitude: Double = 0

    var sessions: [SessionSummary] = []
    var isUploading = false
    var uploadError: String?
    var recorderStatus: String?

    private let motionManager = CMMotionManager()
    private let sensorRecorder = CMSensorRecorder()
    private let updateInterval: TimeInterval = 1.0 / 30.0

    private var csvFileURL: URL?
    private var fileHandle: FileHandle?
    private var recordingStartDate: Date?
    private var startTime: TimeInterval?
    private var buffer: [(t: Double, x: Double, y: Double, z: Double, magnitude: Double)] = []

    // Live WebSocket
    private var liveSocket: URLSessionWebSocketTask?
    private var liveSampleBuffer: [SampleEntry] = []
    private var liveFlushTimer: Timer?

    override init() {
        if let existing = UserDefaults.standard.string(forKey: "userID"), !existing.isEmpty {
            userID = existing
        } else {
            let newID = UUID().uuidString
            UserDefaults.standard.set(newID, forKey: "userID")
            userID = newID
        }
        userName = UserDefaults.standard.string(forKey: "userName") ?? ""
        super.init()
        configureWatchConnectivity()
        restorePendingRecording()
        Task { await recoverFinishedRecordingIfAvailable() }
    }

    func saveName(_ name: String) {
        userName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        UserDefaults.standard.set(userName, forKey: "userName")
    }

    private func configureWatchConnectivity() {
        guard WCSession.isSupported() else { return }
        let session = WCSession.default
        session.delegate = self
        session.activate()
    }

    private func sendUserIDToWatch(_ session: WCSession = .default) {
        let context = ["userID": userID]
        try? session.updateApplicationContext(context)
        if session.isReachable {
            session.sendMessage(context, replyHandler: nil)
        }
    }

    func startUpdates() {
        guard motionManager.isAccelerometerAvailable, !isRecording else { return }
        buffer = []
        uploadError = nil
        recordingStartDate = Date()
        startTime = nil
        elapsedTime = 0
        setupCSVFile()
        startSensorRecording(at: recordingStartDate ?? Date())
        startLiveUpdates()
        startLiveStream()
        isRecording = true
    }

    func stopUpdates() {
        guard isRecording else { return }
        motionManager.stopAccelerometerUpdates()
        isRecording = false
        closeCSVFile()
        stopLiveStream()
        finishSensorRecording(at: Date())
        Task { await self.uploadRecordedSessionOrLiveBuffer() }
    }

    private func restorePendingRecording() {
        guard let startDate = UserDefaults.standard.object(forKey: recordedSessionStartKey) as? Date,
              let endDate = UserDefaults.standard.object(forKey: recordedSessionEndKey) as? Date else { return }
        recordingStartDate = startDate
        if endDate > Date() {
            isRecording = true
            elapsedTime = Date().timeIntervalSince(startDate)
            recorderStatus = "Background recording active."
            startLiveUpdates()
            startLiveStream()
        }
    }

    private func startLiveUpdates() {
        guard motionManager.isAccelerometerAvailable else { return }
        motionManager.accelerometerUpdateInterval = updateInterval
        motionManager.startAccelerometerUpdates(to: .main) { [weak self] data, error in
            guard let self, let data, error == nil else { return }
            self.handleSample(data)
        }
    }

    private func startSensorRecording(at startDate: Date) {
        guard CMSensorRecorder.isAccelerometerRecordingAvailable() else {
            recorderStatus = "Background accelerometer recording is not available on this device."
            return
        }
        UserDefaults.standard.set(startDate, forKey: recordedSessionStartKey)
        UserDefaults.standard.set(startDate.addingTimeInterval(sensorRecorderDuration), forKey: recordedSessionEndKey)
        sensorRecorder.recordAccelerometer(forDuration: sensorRecorderDuration)
        recorderStatus = "Background recording enabled."
    }

    private func finishSensorRecording(at endDate: Date) {
        guard UserDefaults.standard.object(forKey: recordedSessionStartKey) != nil else { return }
        UserDefaults.standard.set(endDate, forKey: recordedSessionEndKey)
    }

    private func handleSample(_ data: CMAccelerometerData) {
        let now = data.timestamp
        if startTime == nil { startTime = now }
        let elapsed = now - (startTime ?? now)

        let x = data.acceleration.x
        let y = data.acceleration.y
        let z = data.acceleration.z
        let mag = sqrt(x * x + y * y + z * z)

        currentX = x; currentY = y; currentZ = z
        currentMagnitude = mag
        elapsedTime = recordingStartDate.map { Date().timeIntervalSince($0) } ?? elapsed

        buffer.append((t: elapsedTime, x: x, y: y, z: z, magnitude: mag))
        writeSampleToCSV(t: elapsedTime, x: x, y: y, z: z, magnitude: mag)
        sendLiveSample(SampleEntry(t: elapsedTime, x: x, y: y, z: z, magnitude: mag))
    }

    private func setupCSVFile() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let url = docs.appendingPathComponent("accel_\(formatter.string(from: Date())).csv")
        csvFileURL = url
        try? "elapsedSeconds,x,y,z,magnitude\n".write(to: url, atomically: true, encoding: .utf8)
        fileHandle = try? FileHandle(forWritingTo: url)
        fileHandle?.seekToEndOfFile()
    }

    private func writeSampleToCSV(t: Double, x: Double, y: Double, z: Double, magnitude: Double) {
        if let data = "\(t),\(x),\(y),\(z),\(magnitude)\n".data(using: .utf8) {
            fileHandle?.write(data)
        }
    }

    private func closeCSVFile() {
        fileHandle?.closeFile()
        fileHandle = nil
    }

    // MARK: - Live WebSocket

    private func startLiveStream() {
        guard let url = URL(string: "\(baseURL.replacingOccurrences(of: "https://", with: "wss://"))/ws/ingest/\(userID)") else { return }
        liveSocket = URLSession.shared.webSocketTask(with: url)
        liveSocket?.resume()
        liveFlushTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.flushLiveBuffer()
        }
    }

    private func stopLiveStream() {
        liveFlushTimer?.invalidate()
        liveFlushTimer = nil
        flushLiveBuffer()
        liveSocket?.cancel(with: .goingAway, reason: nil)
        liveSocket = nil
    }

    private func sendLiveSample(_ sample: SampleEntry) {
        liveSampleBuffer.append(sample)
    }

    private func flushLiveBuffer() {
        guard !liveSampleBuffer.isEmpty, let socket = liveSocket else { return }
        let batch = liveSampleBuffer
        liveSampleBuffer = []
        struct Payload: Encodable { let samples: [SampleEntry] }
        guard let data = try? JSONEncoder().encode(Payload(samples: batch)),
              let json = String(data: data, encoding: .utf8) else { return }
        socket.send(.string(json)) { _ in }
    }

    // MARK: - Upload

    private func uploadRecordedSessionOrLiveBuffer() async {
        if await uploadRecordedSessionIfAvailable() {
            buffer = []
            return
        }
        await uploadInChunks(samples: buffer, startedAt: recordingStartDate ?? Date())
    }

    private func uploadRecordedSessionIfAvailable() async -> Bool {
        guard let startedAt = UserDefaults.standard.object(forKey: recordedSessionStartKey) as? Date,
              let requestedEnd = UserDefaults.standard.object(forKey: recordedSessionEndKey) as? Date else { return false }
        let endedAt = min(requestedEnd, Date())
        guard endedAt > startedAt else { return false }
        guard let samples = recordedSamples(from: startedAt, to: endedAt), !samples.isEmpty else { return false }
        await uploadInChunks(samples: samples, startedAt: startedAt)
        clearPendingRecordedSession()
        return true
    }

    private func uploadInChunks(
        samples: [(t: Double, x: Double, y: Double, z: Double, magnitude: Double)],
        startedAt: Date
    ) async {
        let chunkDuration: Double = 3600
        var chunks: [( [(t: Double, x: Double, y: Double, z: Double, magnitude: Double)], Date )] = []
        var chunkSamples: [(t: Double, x: Double, y: Double, z: Double, magnitude: Double)] = []
        var chunkOriginT: Double = samples.first?.t ?? 0
        var chunkStartDate: Date = startedAt

        for sample in samples {
            if sample.t - chunkOriginT >= chunkDuration && !chunkSamples.isEmpty {
                chunks.append((chunkSamples, chunkStartDate))
                chunkOriginT = sample.t
                chunkStartDate = startedAt.addingTimeInterval(sample.t)
                chunkSamples = []
            }
            chunkSamples.append(sample)
        }
        if !chunkSamples.isEmpty { chunks.append((chunkSamples, chunkStartDate)) }

        let total = chunks.count
        for (i, (chunk, chunkStart)) in chunks.enumerated() {
            await MainActor.run {
                recorderStatus = total > 1 ? "Uploading hour \(i + 1) of \(total)..." : "Uploading session..."
            }
            let payloadSamples = chunk.map { SampleEntry(t: $0.t, x: $0.x, y: $0.y, z: $0.z, magnitude: $0.magnitude) }
            let isoFormatter = ISO8601DateFormatter()
            isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            _ = await uploadSession(startedAt: isoFormatter.string(from: chunkStart), samples: payloadSamples)
        }

        await fetchSessions()
        await MainActor.run { recorderStatus = total > 1 ? "Uploaded \(total) hours of data." : nil }
    }

    private func recoverFinishedRecordingIfAvailable() async {
        guard let endDate = UserDefaults.standard.object(forKey: recordedSessionEndKey) as? Date,
              endDate <= Date() else { return }
        await MainActor.run {
            isRecording = false
            motionManager.stopAccelerometerUpdates()
            closeCSVFile()
        }
        if !(await uploadRecordedSessionIfAvailable()) {
            await MainActor.run { recorderStatus = "Recorded data not ready yet. Reopen the app in a few minutes." }
        }
    }

    private func recordedSamples(from startedAt: Date, to endedAt: Date) -> [(t: Double, x: Double, y: Double, z: Double, magnitude: Double)]? {
        guard let dataList = sensorRecorder.accelerometerData(from: startedAt, to: endedAt) else { return nil }
        var samples: [(t: Double, x: Double, y: Double, z: Double, magnitude: Double)] = []
        var iterator = NSFastEnumerationIterator(dataList)
        while let data = iterator.next() as? CMRecordedAccelerometerData {
            let x = data.acceleration.x
            let y = data.acceleration.y
            let z = data.acceleration.z
            let magnitude = sqrt(x * x + y * y + z * z)
            let elapsed = data.startDate.timeIntervalSince(startedAt)
            samples.append((t: elapsed, x: x, y: y, z: z, magnitude: magnitude))
        }
        return samples
    }

    private func clearPendingRecordedSession() {
        UserDefaults.standard.removeObject(forKey: recordedSessionStartKey)
        UserDefaults.standard.removeObject(forKey: recordedSessionEndKey)
    }

    private func uploadSession(startedAt: String, samples: [SampleEntry]) async -> Bool {
        guard !samples.isEmpty else { return false }
        await MainActor.run { isUploading = true; uploadError = nil }
        guard let url = URL(string: "\(baseURL)/sessions") else {
            await MainActor.run { isUploading = false; uploadError = "Invalid server URL" }
            return false
        }
        let body = SessionUploadPayload(userID: userID, userName: userName, startedAt: startedAt, samples: samples)
        do {
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONEncoder().encode(body)
            let (_, response) = try await URLSession.shared.data(for: request)
            if let r = response as? HTTPURLResponse, r.statusCode >= 300 { throw URLError(.badServerResponse) }
            await MainActor.run { isUploading = false }
            return true
        } catch {
            await MainActor.run { isUploading = false; uploadError = "Upload failed: \(error.localizedDescription)" }
            return false
        }
    }

    private func receiveWatchSessionFile(_ fileURL: URL) async {
        await MainActor.run { isUploading = true; uploadError = nil }
        do {
            let data = try Data(contentsOf: fileURL)
            let payload = try JSONDecoder().decode(WatchSessionTransferPayload.self, from: data)
            _ = await uploadSession(startedAt: payload.startedAt, samples: payload.samples)
        } catch {
            await MainActor.run {
                isUploading = false
                uploadError = "Could not import watch recording: \(error.localizedDescription)"
            }
        }
    }

    func fetchSessions() async {
        guard let url = URL(string: "\(baseURL)/users/\(userID)/sessions") else { return }
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            let decoded = try JSONDecoder().decode([SessionSummary].self, from: data)
            await MainActor.run { sessions = decoded }
        } catch {
            print("fetchSessions error: \(error)")
        }
    }
}

// MARK: - WCSessionDelegate

extension AccelerometerManager {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        guard activationState == .activated else { return }
        sendUserIDToWatch(session)
    }

    func session(_ session: WCSession, didReceiveMessage message: [String: Any], replyHandler: @escaping ([String: Any]) -> Void) {
        if message["request"] as? String == "userID" {
            replyHandler(["userID": userID])
        } else {
            replyHandler([:])
        }
    }

    func session(_ session: WCSession, didReceive file: WCSessionFile) {
        guard file.metadata?["kind"] as? String == "watchSessionUpload" else { return }
        Task { await receiveWatchSessionFile(file.fileURL) }
    }

    func sessionDidBecomeInactive(_ session: WCSession) {}
    func sessionDidDeactivate(_ session: WCSession) { session.activate() }
}

// MARK: - OnboardingView

struct OnboardingView: View {
    @Bindable var manager: AccelerometerManager
    @State private var nameInput = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(spacing: 0) {
            Spacer()
            VStack(spacing: 16) {
                Image(systemName: "waveform.circle.fill")
                    .font(.system(size: 72))
                    .foregroundStyle(.blue)
                Text("NeuroMotion")
                    .font(.system(size: 34, weight: .bold, design: .rounded))
                Text("Movement tracking for neurological health")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            Spacer()
            VStack(alignment: .leading, spacing: 8) {
                Text("Your Name")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 4)
                TextField("Enter your full name", text: $nameInput)
                    .font(.body)
                    .padding()
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
                    .focused($focused)
                    .submitLabel(.go)
                    .onSubmit { confirmName() }
            }
            .padding(.horizontal)
            Button(action: confirmName) {
                Text("Get Started")
                    .font(.headline)
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(
                        nameInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Color.gray : Color.blue,
                        in: RoundedRectangle(cornerRadius: 14)
                    )
            }
            .disabled(nameInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            .padding(.horizontal)
            .padding(.top, 12)
            Spacer().frame(height: 48)
        }
        .onAppear { focused = true }
    }

    private func confirmName() {
        let trimmed = nameInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        manager.saveName(trimmed)
    }
}

// MARK: - AccelerometerView

struct AccelerometerView: View {
    @Bindable var manager: AccelerometerManager

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                VStack(spacing: 8) {
                    Text("Live Accelerometer").font(.headline)
                    HStack(spacing: 20) {
                        axisCard("X", value: manager.currentX, color: .red)
                        axisCard("Y", value: manager.currentY, color: .green)
                        axisCard("Z", value: manager.currentZ, color: .blue)
                    }
                    HStack {
                        Image(systemName: "waveform.path")
                        Text(String(format: "Magnitude: %.4f g", manager.currentMagnitude)).monospacedDigit()
                    }.foregroundStyle(.purple)
                    Text(String(format: "Elapsed: %.1f s", manager.elapsedTime))
                        .foregroundStyle(.secondary).monospacedDigit()
                }
                .padding()
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))

                if manager.isUploading {
                    HStack { ProgressView(); Text("Uploading session...").foregroundStyle(.secondary) }
                } else if let err = manager.uploadError {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                        Text(err).font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.leading)
                    }.padding(.horizontal)
                } else if let status = manager.recorderStatus {
                    HStack {
                        Image(systemName: "recordingtape").foregroundStyle(.secondary)
                        Text(status).font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.leading)
                    }.padding(.horizontal)
                }

                Button(action: { manager.isRecording ? manager.stopUpdates() : manager.startUpdates() }) {
                    Label(
                        manager.isRecording ? "Stop Recording" : "Start Recording",
                        systemImage: manager.isRecording ? "stop.circle.fill" : "record.circle"
                    )
                    .font(.title3.bold()).foregroundStyle(.white).padding()
                    .frame(maxWidth: .infinity)
                    .background(manager.isRecording ? Color.red : Color.accentColor,
                                in: RoundedRectangle(cornerRadius: 14))
                }.padding(.horizontal)

                Spacer()
            }
            .padding()
            .navigationTitle(manager.userName)
        }
    }

    private func axisCard(_ label: String, value: Double, color: Color) -> some View {
        VStack(spacing: 4) {
            Text(label).font(.caption.bold()).foregroundStyle(color)
            Text(String(format: "%.4f", value)).font(.caption2).monospacedDigit()
        }
        .frame(minWidth: 60).padding(8)
        .background(color.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - SessionHistoryView

struct SessionHistoryView: View {
    @Bindable var manager: AccelerometerManager

    var body: some View {
        NavigationStack {
            Group {
                if manager.sessions.isEmpty {
                    ContentUnavailableView("No Sessions Yet", systemImage: "waveform",
                        description: Text("Record and upload your first session to see history."))
                } else {
                    List(manager.sessions) { session in
                        NavigationLink(destination: SessionDetailView(session: session)) {
                            SessionRowView(session: session)
                        }
                    }
                }
            }
            .navigationTitle("History")
            .refreshable { await manager.fetchSessions() }
        }
    }
}

// MARK: - SessionRowView

struct SessionRowView: View {
    let session: SessionSummary

    private var formattedDate: String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds,
                           .withDashSeparatorInDate, .withColonSeparatorInTime]
        guard let date = f.date(from: session.startedAt) else { return session.startedAt }
        let d = DateFormatter(); d.dateStyle = .medium; d.timeStyle = .short
        return d.string(from: date)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(formattedDate).font(.subheadline.bold())
            HStack {
                Text(String(format: "%.1f s  •  %d samples", session.durationS, session.sampleCount))
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                if let a = session.analysis {
                    if let tau = a.tau {
                        Label(String(format: "τ=%.2f", tau), systemImage: "chart.line.downtrend.xyaxis")
                            .font(.caption.bold())
                            .foregroundStyle(a.isScaleFree == true ? .green : .orange)
                    } else {
                        Text(a.error != nil ? "No fit" : "Pending")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                } else {
                    Text("No analysis").font(.caption).foregroundStyle(.tertiary)
                }
            }
        }.padding(.vertical, 2)
    }
}

// MARK: - SessionDetailView

struct SessionDetailView: View {
    let session: SessionSummary

    private var formattedDate: String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds,
                           .withDashSeparatorInDate, .withColonSeparatorInTime]
        guard let date = f.date(from: session.startedAt) else { return session.startedAt }
        let d = DateFormatter(); d.dateStyle = .long; d.timeStyle = .medium
        return d.string(from: date)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(formattedDate).font(.subheadline).foregroundStyle(.secondary)
                    Text(String(format: "Duration: %.1f s  •  %d samples", session.durationS, session.sampleCount))
                        .font(.caption).foregroundStyle(.secondary)
                }
                Divider()

                if let a = session.analysis, a.tau != nil {
                    HStack {
                        Image(systemName: a.isScaleFree == true ? "checkmark.seal.fill" : "xmark.seal.fill")
                            .foregroundStyle(a.isScaleFree == true ? .green : .red)
                            .font(.title2)
                        Text(a.isScaleFree == true ? "Scale-Free Dynamics" : "Not Scale-Free")
                            .font(.title3.bold())
                            .foregroundStyle(a.isScaleFree == true ? .green : .red)
                        Spacer()
                    }
                    .padding()
                    .background((a.isScaleFree == true ? Color.green : Color.red).opacity(0.1),
                                in: RoundedRectangle(cornerRadius: 12))

                    VStack(spacing: 12) {
                        VStack(spacing: 4) {
                            Text("Power Law Exponent").font(.subheadline).foregroundStyle(.secondary)
                            Text(String(format: "τ = %.3f", a.tau!))
                                .font(.system(size: 48, weight: .bold, design: .rounded))
                            Text("Ideal range: 1.5 – 2.5").font(.caption).foregroundStyle(.secondary)
                        }
                        Divider()
                        if let gof = a.goodnessOfFit {
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text("Goodness of Fit").font(.subheadline.bold())
                                    Spacer()
                                    Text(String(format: "%.1f%%", gof * 100))
                                        .font(.subheadline.bold())
                                        .foregroundStyle(gof >= 0.8 ? .green : (gof >= 0.6 ? .yellow : .red))
                                }
                                ProgressView(value: gof)
                                    .tint(gof >= 0.8 ? .green : (gof >= 0.6 ? .yellow : .red))
                                Text("≥ 80% required for scale-free classification")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        Divider()
                        HStack(spacing: 20) {
                            if let plr = a.powerLawRange {
                                statBox(label: "Fit Range", value: String(format: "%.2f dec", plr))
                            }
                            if let n = a.nEvents {
                                statBox(label: "Events", value: "\(n)")
                            }
                        }
                    }
                    .padding()
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Interpretation").font(.subheadline.bold())
                        Text("τ ≈ 1.5    Branching ratio near critical point")
                        Text("τ ≈ 2.0    Mean-field critical exponent")
                        Text("τ < 1.5    Sub-critical, reduced complexity")
                        Text("τ > 2.5    Super-critical, excessive synchrony")
                        Text("")
                        Text("Scale-free movement dynamics (goodness of fit ≥ 80%) indicate healthy neurological function.")
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .font(.caption).foregroundStyle(.secondary)
                    .padding()
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))

                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "chart.bar.xaxis").font(.largeTitle).foregroundStyle(.secondary)
                        Text("No Analysis Available").font(.headline)
                        if let a = session.analysis, let err = a.error {
                            Text(err).font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
                        } else {
                            Text("Too few samples or events detected.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .frame(maxWidth: .infinity).padding(40)
                }
            }.padding()
        }
        .navigationTitle("Session Detail")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func statBox(label: String, value: String) -> some View {
        VStack(spacing: 4) {
            Text(value).font(.title3.bold())
            Text(label).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity).padding(10)
        .background(Color.secondary.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - ContentView

struct ContentView: View {
    @State private var manager = AccelerometerManager()

    var body: some View {
        if manager.userName.isEmpty {
            OnboardingView(manager: manager)
        } else {
            TabView {
                AccelerometerView(manager: manager)
                    .tabItem { Label("Record", systemImage: "waveform.circle") }
                SessionHistoryView(manager: manager)
                    .tabItem { Label("History", systemImage: "clock.arrow.circlepath") }
                    .onAppear { Task { await manager.fetchSessions() } }
            }
        }
    }
}

@main
struct AccelerometerApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

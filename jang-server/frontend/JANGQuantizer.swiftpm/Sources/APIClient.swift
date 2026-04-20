import Foundation

@Observable
class APIClient {
    var baseURL = "http://localhost:8420"
    var apiKey = ""

    private var decoder: JSONDecoder {
        let d = JSONDecoder()
        return d
    }

    private func request(_ method: String, _ path: String, body: Data? = nil) async throws -> Data {
        guard let url = URL(string: "\(baseURL)\(path)") else {
            throw APIError.invalidURL
        }
        var req = URLRequest(url: url)
        req.httpMethod = method
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !apiKey.isEmpty {
            req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        req.httpBody = body
        let (data, response) = try await URLSession.shared.data(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw APIError.unknown
        }
        if http.statusCode >= 400 {
            let msg = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw APIError.http(http.statusCode, msg)
        }
        return data
    }

    // MARK: - Endpoints

    func submitJob(modelId: String, profile: String, user: String, priority: Int = 0) async throws -> JobResponse {
        let body = JobRequest(modelId: modelId, profile: profile, user: user, priority: priority, webhookUrl: "")
        let data = try JSONEncoder().encode(body)
        let result = try await request("POST", "/jobs", body: data)
        return try decoder.decode(JobResponse.self, from: result)
    }

    func getJob(_ jobId: String) async throws -> JobResponse {
        let data = try await request("GET", "/jobs/\(jobId)")
        return try decoder.decode(JobResponse.self, from: data)
    }

    func listJobs(user: String? = nil, phase: String? = nil) async throws -> [JobResponse] {
        // M185 (iter 120): pre-fix this method splice'd `user` and `phase`
        // values directly into the query string — a username containing
        // `&`, `=`, `?`, `#`, `+`, space, or `%` would break the URL OR
        // inject extra parameters. A user named `alice&phase=COMPLETED`
        // would override the phase filter. Build with URLComponents +
        // URLQueryItem so URLSession encodes correctly.
        var components = URLComponents()
        components.path = "/jobs"
        var items: [URLQueryItem] = []
        if let u = user, !u.isEmpty { items.append(URLQueryItem(name: "user", value: u)) }
        if let p = phase, !p.isEmpty { items.append(URLQueryItem(name: "phase", value: p)) }
        if !items.isEmpty { components.queryItems = items }
        // .url returns nil if components don't compose a valid URL; for a
        // path-only relative URL this should always succeed, but guard
        // defensively to surface failures clearly instead of silently
        // hitting the wrong endpoint.
        guard let pathPlusQuery = components.string else {
            throw APIError.invalidURL
        }
        let data = try await request("GET", pathPlusQuery)
        return try decoder.decode([JobResponse].self, from: data)
    }

    func getQueue() async throws -> QueueResponse {
        let data = try await request("GET", "/queue")
        return try decoder.decode(QueueResponse.self, from: data)
    }

    func cancelJob(_ jobId: String) async throws {
        _ = try await request("DELETE", "/jobs/\(jobId)")
    }

    func retryJob(_ jobId: String) async throws -> JobResponse {
        let data = try await request("POST", "/jobs/\(jobId)/retry")
        // The response has a "new_job" key
        struct RetryResponse: Codable {
            let message: String
            let newJob: JobResponse
            enum CodingKeys: String, CodingKey {
                case message
                case newJob = "new_job"
            }
        }
        let resp = try decoder.decode(RetryResponse.self, from: data)
        return resp.newJob
    }

    func getJobLogs(_ jobId: String) async throws -> [String] {
        let data = try await request("GET", "/jobs/\(jobId)/logs")
        struct LogResponse: Codable {
            let lines: [String]
        }
        return try decoder.decode(LogResponse.self, from: data).lines
    }

    func getHealth() async throws -> HealthResponse {
        let data = try await request("GET", "/health")
        return try decoder.decode(HealthResponse.self, from: data)
    }
}

enum APIError: LocalizedError {
    case invalidURL
    case http(Int, String)
    case unknown

    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid URL"
        case .http(let code, let msg): return "HTTP \(code): \(msg)"
        case .unknown: return "Unknown error"
        }
    }
}

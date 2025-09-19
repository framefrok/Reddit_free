# re_dpi.py — v3.1: Domain Fronting + Adaptive Obfuscation for Reddit — FIXED
# Bypasses DPI by disguising as Cloudflare-legitimate traffic
# Works in Termux, Android, without root

import asyncio
import ssl
import socket
import random
import time
import logging
import argparse
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("re_dpi.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("re_dpi")

class ObfuscationMode(Enum):
    TCP_SEGMENTATION = "tcp_segmentation"
    DELAY_JITTER = "delay_jitter"
    DOMAIN_FRONTING = "domain_fronting"
    USER_AGENT_ROTATION = "user_agent_rotation"
    HEADER_OBFUSCATION = "header_obfuscation"

@dataclass
class ConnectionProfile:
    delay_jitter_ms: Tuple[int, int] = (1, 15)
    packet_burst_size: int = 3
    fronting_host: str = "cloudflare.com"
    obfuscation_modes: List[ObfuscationMode] = None

    def __post_init__(self):
        if self.obfuscation_modes is None:
            self.obfuscation_modes = [
                ObfuscationMode.TCP_SEGMENTATION,
                ObfuscationMode.DELAY_JITTER,
                ObfuscationMode.DOMAIN_FRONTING,
                ObfuscationMode.USER_AGENT_ROTATION
            ]

class SmartParameterOptimizer:
    def __init__(self):
        self.success_rates: Dict[str, float] = {}
        self.latency_profiles: Dict[str, List[float]] = {}
        self.last_optimization = 0
        self.optimization_interval = 300

    def record_attempt(self, profile_hash: str, success: bool, latency: float):
        if profile_hash not in self.success_rates:
            self.success_rates[profile_hash] = 0.0
            self.latency_profiles[profile_hash] = []

        alpha = 0.3
        current_rate = self.success_rates[profile_hash]
        self.success_rates[profile_hash] = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate

        self.latency_profiles[profile_hash].append(latency)
        if len(self.latency_profiles[profile_hash]) > 10:
            self.latency_profiles[profile_hash].pop(0)

    def should_optimize(self) -> bool:
        return time.time() - self.last_optimization > self.optimization_interval

    def generate_optimized_profile(self, current_profile: ConnectionProfile) -> ConnectionProfile:
        if not self.should_optimize():
            return current_profile

        logger.info("Optimizing evasion parameters based on network feedback")

        variations = []
        fronting_candidates = [
            "cloudflare.com",
            "cdnjs.com",
            "1.1.1.1",
            "workers.dev",
            "pages.dev",
            "discord.com",
            "telegram.org"
        ]

        for i in range(5):
            variant = ConnectionProfile()
            variant.delay_jitter_ms = (
                random.randint(0, 5),
                random.randint(10, 50)
            )
            variant.packet_burst_size = random.randint(1, 5)
            variant.fronting_host = random.choice(fronting_candidates)
            variant.obfuscation_modes = random.sample(
                list(ObfuscationMode),
                k=random.randint(2, 4)
            )
            variations.append(variant)

        scored_variations = []
        for variant in variations:
            variant_hash = self._profile_hash(variant)
            success_rate = self.success_rates.get(variant_hash, 0.5)
            avg_latency = sum(self.latency_profiles.get(variant_hash, [1000])) / len(self.latency_profiles.get(variant_hash, [1]))
            score = success_rate * (1000.0 / max(avg_latency, 1))
            scored_variations.append((variant, score))

        best_variant, best_score = max(scored_variations, key=lambda x: x[1])
        base_hash = self._profile_hash(current_profile)
        current_score = self.success_rates.get(base_hash, 0.5) * (1000.0 / max(
            sum(self.latency_profiles.get(base_hash, [1000])) / len(self.latency_profiles.get(base_hash, [1])), 1
        ))

        if best_score > current_score * 1.2:
            logger.info(f"🚀 Switching to optimized profile: fronting={best_variant.fronting_host}, score={best_score:.2f}")
            self.last_optimization = time.time()
            return best_variant
        else:
            logger.info("✅ Current profile remains optimal")
            return current_profile

    def _profile_hash(self, profile: ConnectionProfile) -> str:
        profile_dict = {
            "delay_jitter": profile.delay_jitter_ms,
            "burst_size": profile.packet_burst_size,
            "fronting": profile.fronting_host,
            "modes": [mode.value for mode in profile.obfuscation_modes]
        }
        return hashlib.md5(json.dumps(profile_dict, sort_keys=True).encode()).hexdigest()

class TCPObfuscator:
    def __init__(self, profile: ConnectionProfile):
        self.profile = profile

    async def send_with_obfuscation(self, writer: asyncio.StreamWriter, data: bytes):
        if ObfuscationMode.TCP_SEGMENTATION in self.profile.obfuscation_modes:
            await self._send_with_segmentation(writer, data)
        else:
            await self._send_with_jitter(writer, data)

    async def _send_with_segmentation(self, writer: asyncio.StreamWriter, data: bytes):
        segment_size = random.choice([512, 1024, 1200, 1400])
        burst_size = self.profile.packet_burst_size
        min_delay, max_delay = self.profile.delay_jitter_ms

        for i in range(0, len(data), segment_size * burst_size):
            burst = data[i:i + segment_size * burst_size]

            for j in range(0, len(burst), segment_size):
                segment = burst[j:j + segment_size]
                writer.write(segment)
                await writer.drain()

                if j + segment_size < len(burst):
                    await asyncio.sleep(random.uniform(0.001, 0.008))

            if i + segment_size * burst_size < len(data):
                jitter_delay = random.uniform(min_delay / 1000.0, max_delay / 1000.0)
                await asyncio.sleep(jitter_delay)

    async def _send_with_jitter(self, writer: asyncio.StreamWriter, data: bytes):
        writer.write(data)
        await writer.drain()
        min_delay, max_delay = self.profile.delay_jitter_ms
        jitter_delay = random.uniform(min_delay / 1000.0, max_delay / 1000.0)
        await asyncio.sleep(jitter_delay)

class RedditDPIBypasser:
    def __init__(self, target_host: str = "www.reddit.com", target_port: int = 443):
        self.target_host = target_host
        self.target_port = target_port
        self.optimizer = SmartParameterOptimizer()
        self.current_profile = ConnectionProfile()
        self.session_count = 0
        self.successful_sessions = 0

    async def make_request(self, path: str = "/", headers: Optional[Dict[str, str]] = None) -> Optional[str]:
        start_time = time.time()
        profile_hash = hashlib.md5(str(self.current_profile.__dict__).encode()).hexdigest()

        if self.optimizer.should_optimize():
            self.current_profile = self.optimizer.generate_optimized_profile(self.current_profile)

        try:
            fronting_host = self.current_profile.fronting_host
            actual_host = self.target_host

            logger.info(f"📡 Connecting via fronting: TLS={fronting_host} | HTTP-Host={actual_host}")

            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            ssl_context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20")
            ssl_context.set_alpn_protocols(["http/1.1"])

            addr_info = await asyncio.get_event_loop().getaddrinfo(
                actual_host, self.target_port,
                family=socket.AF_INET, type=socket.SOCK_STREAM
            )
            target_ip = addr_info[0][4][0]

            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    target_ip,
                    self.target_port,
                    ssl=ssl_context,
                    server_hostname=fronting_host
                ),
                timeout=15.0
            )

            if headers is None:
                headers = self._build_headers(actual_host)

            request_lines = [f"GET {path} HTTP/1.1", f"Host: {actual_host}"]
            for key, value in headers.items():
                request_lines.append(f"{key}: {value}")
            request_lines.extend(["", ""])
            request = "\r\n".join(request_lines).encode()

            tcp_obfuscator = TCPObfuscator(self.current_profile)
            await tcp_obfuscator.send_with_obfuscation(writer, request)

            response = await asyncio.wait_for(self._read_response(reader), timeout=20.0)

            latency = time.time() - start_time
            self.optimizer.record_attempt(profile_hash, True, latency)
            self.successful_sessions += 1
            self.session_count += 1

            writer.close()
            await writer.wait_closed()

            return response

        except ssl.SSLError as e:
            if "APPLICATION_DATA_AFTER_CLOSE_NOTIFY" in str(e):
                logger.warning("⚠️  DPI detected: connection terminated by middlebox")
            else:
                logger.error(f"SSL Error: {e}")
        except asyncio.TimeoutError:
            logger.error("⏱️  Timeout: DPI may be throttling or dropping packets")
        except Exception as e:
            logger.error(f"Request failed: {e}")

        latency = time.time() - start_time
        self.optimizer.record_attempt(profile_hash, False, latency)
        self.session_count += 1
        return None

    def _build_headers(self, host: str) -> Dict[str, str]:
        headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }

        if ObfuscationMode.HEADER_OBFUSCATION in self.current_profile.obfuscation_modes:
            noise_headers = {
                "X-Forwarded-For": f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
                "X-Real-IP": f"172.{random.randint(16,31)}.{random.randint(0,255)}.{random.randint(1,254)}",
                "X-Client-IP": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
                "X-Originating-IP": f"192.168.{random.randint(0,255)}.{random.randint(1,254)}",
                "Via": f"1.1 Chrome-Compression-Proxy, 1.1 {host}"
            }
            headers.update(noise_headers)

        return headers

    def _get_random_user_agent(self) -> str:
        if ObfuscationMode.USER_AGENT_ROTATION not in self.current_profile.obfuscation_modes:
            return "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

        agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Linux; Android 14; SM-S911B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ]
        return random.choice(agents)

    async def _read_response(self, reader: asyncio.StreamReader) -> str:
        response_data = b""
        content_length = None
        chunked = False
        headers_done = False

        while True:
            chunk = await reader.read(4096)
            if not chunk:
                break
            response_data += chunk

            if not headers_done and b"\r\n\r\n" in response_data:
                headers_done = True
                header_bytes = response_data.split(b"\r\n\r\n", 1)[0]
                header_text = header_bytes.decode('utf-8', errors='ignore').lower()
                if "content-length:" in header_text:
                    try:
                        for line in header_text.split("\r\n"):
                            if line.startswith("content-length:"):
                                content_length = int(line.split(":", 1)[1].strip())
                                break
                    except:
                        pass
                if "transfer-encoding: chunked" in header_text:
                    chunked = True

            if len(response_data) > 50000:
                break

            if content_length and len(response_data) >= content_length + len(header_bytes) + 4:
                break

        return response_data.decode('utf-8', errors='ignore')

    def get_statistics(self) -> Dict[str, Any]:
        success_rate = self.successful_sessions / max(self.session_count, 1)
        return {
            "total_sessions": self.session_count,
            "successful_sessions": self.successful_sessions,
            "success_rate": round(success_rate, 3),
            "current_profile": {
                "modes": [m.value for m in self.current_profile.obfuscation_modes],
                "jitter_ms": self.current_profile.delay_jitter_ms,
                "burst_size": self.current_profile.packet_burst_size,
                "fronting_host": self.current_profile.fronting_host
            }
        }

async def main():
    parser = argparse.ArgumentParser(description="Reddit DPI Bypasser v3.1 — Domain Fronting FIXED")
    parser.add_argument("--host", default="www.reddit.com", help="Target host (default: www.reddit.com)")
    parser.add_argument("--port", type=int, default=443, help="Target port (default: 443)")
    parser.add_argument("--path", default="/", help="Request path (default: /)")
    parser.add_argument("--test", action="store_true", help="Run connectivity test")
    parser.add_argument("--stats", action="store_true", help="Show statistics")

    args = parser.parse_args()

    bypasser = RedditDPIBypasser(args.host, args.port)

    if args.test:
        logger.info(f"🧪 Testing connection to {args.host}...")
        response = await bypasser.make_request(args.path)
        if response:
            logger.info("✅ Success! Received response from Reddit.")
            lines = response.split("\n")
            for line in lines[:10]:
                if line.startswith("HTTP/") or "<title>" in line:
                    print(f"  {line.strip()}")
            print(f"\n📦 Response size: {len(response)} bytes")
        else:
            logger.error("❌ Failed to connect. Try again — optimizer will adjust parameters.")
            sys.exit(1)
    elif args.stats:
        logger.info("📈 Running 3 test requests for statistics...")
        for i in range(3):
            logger.info(f"  Request {i+1}/3...")
            await bypasser.make_request("/")
            await asyncio.sleep(1)
        stats = bypasser.get_statistics()
        print("\n📊 Final Statistics:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print("🚀 Reddit DPI Bypasser — Type 'quit' to exit")
        while True:
            try:
                path = input("\nEnter Reddit path (e.g. /r/Python): ").strip()
                if path.lower() in ('quit', 'exit', 'q'):
                    break
                if not path:
                    path = "/"
                print(f"➡️  Requesting: {path}")
                response = await bypasser.make_request(path)
                if response:
                    print(f"✅ Success! Received {len(response)} bytes")
                    first_line = response.split("\n")[0] if "\n" in response else response[:100]
                    print(f"HeaderCode: {first_line}")
                    if "<title>" in response:
                        title_start = response.find("<title>") + 7
                        title_end = response.find("</title>", title_start)
                        if title_start > 7 and title_end > title_start:
                            title = response[title_start:title_end].strip()
                            print(f"📄 Title: {title}")
                else:
                    print("❌ Request failed — optimizer will try new parameters next time")
            except KeyboardInterrupt:
                print("\n🛑 Exiting...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

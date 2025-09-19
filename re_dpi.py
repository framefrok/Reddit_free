# re_dpi.py — Advanced DPI circumvention tool for Reddit access on Android/Termux
# Designed for robust evasion of Deep Packet Inspection systems
# Author: Advanced Network Obfuscation Engineer
# License: MIT

import asyncio
import ssl
import socket
import struct
import random
import time
import logging
import argparse
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import os
import sys
import selectors
import threading
import json
import base64

# Configure logging
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
    TLS_FRAGMENTATION = "tls_fragmentation"
    HTTP2_PSEUDO_HEADERS = "http2_pseudo_headers"
    TCP_SEGMENTATION = "tcp_segmentation"
    TLS_GREASE = "tls_grease"
    TLS_RECORD_SPLITTING = "tls_record_splitting"
    CUSTOM_SNI = "custom_sni"
    TLS_PADDING = "tls_padding"

@dataclass
class ConnectionProfile:
    fragmentation_size: int = 1300
    delay_jitter_ms: Tuple[int, int] = (1, 15)
    packet_burst_size: int = 3
    tls_record_split_size: int = 5
    enable_grease: bool = True
    custom_sni: Optional[str] = None
    padding_strategy: str = "random"
    obfuscation_modes: List[ObfuscationMode] = None

    def __post_init__(self):
        if self.obfuscation_modes is None:
            self.obfuscation_modes = [
                ObfuscationMode.TLS_FRAGMENTATION,
                ObfuscationMode.TCP_SEGMENTATION,
                ObfuscationMode.TLS_GREASE,
                ObfuscationMode.TLS_RECORD_SPLITTING
            ]

class SmartParameterOptimizer:
    """Dynamically adjusts evasion parameters based on network feedback"""

    def __init__(self):
        self.success_rates: Dict[str, float] = {}
        self.latency_profiles: Dict[str, List[float]] = {}
        self.last_optimization = 0
        self.optimization_interval = 300  # 5 minutes

    def record_attempt(self, profile_hash: str, success: bool, latency: float):
        if profile_hash not in self.success_rates:
            self.success_rates[profile_hash] = 0.0
            self.latency_profiles[profile_hash] = []

        # Exponential moving average for success rate
        alpha = 0.3
        current_rate = self.success_rates[profile_hash]
        self.success_rates[profile_hash] = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
        
        # Keep last 10 latency measurements
        self.latency_profiles[profile_hash].append(latency)
        if len(self.latency_profiles[profile_hash]) > 10:
            self.latency_profiles[profile_hash].pop(0)

    def should_optimize(self) -> bool:
        return time.time() - self.last_optimization > self.optimization_interval

    def generate_optimized_profile(self, current_profile: ConnectionProfile) -> ConnectionProfile:
        if not self.should_optimize():
            return current_profile

        logger.info("Optimizing evasion parameters based on network feedback")

        # Create variations
        variations = []
        base_hash = self._profile_hash(current_profile)

        for i in range(5):
            variant = ConnectionProfile()
            variant.fragmentation_size = random.choice([1200, 1300, 1400, 1460])
            variant.delay_jitter_ms = (
                random.randint(0, 5),
                random.randint(10, 50)
            )
            variant.packet_burst_size = random.randint(1, 5)
            variant.tls_record_split_size = random.randint(3, 8)
            variant.enable_grease = random.choice([True, False])
            variant.custom_sni = random.choice([
                None,
                "www.google.com",
                "cloudflare.com",
                "wikipedia.org"
            ])
            variant.padding_strategy = random.choice(["random", "fixed", "none"])
            variant.obfuscation_modes = random.sample(
                list(ObfuscationMode),
                k=random.randint(2, 5)
            )
            variations.append(variant)

        # Score each variation
        scored_variations = []
        for variant in variations:
            variant_hash = self._profile_hash(variant)
            success_rate = self.success_rates.get(variant_hash, 0.5)
            avg_latency = sum(self.latency_profiles.get(variant_hash, [1000])) / len(self.latency_profiles.get(variant_hash, [1]))
            
            # Score: prioritize success rate, penalize high latency
            score = success_rate * (1000.0 / max(avg_latency, 1))
            scored_variations.append((variant, score))

        # Return best variation or original if all worse
        best_variant, best_score = max(scored_variations, key=lambda x: x[1])
        current_score = self.success_rates.get(base_hash, 0.5) * (1000.0 / max(
            sum(self.latency_profiles.get(base_hash, [1000])) / len(self.latency_profiles.get(base_hash, [1])), 1
        ))

        if best_score > current_score * 1.2:  # 20% improvement threshold
            logger.info(f"Switching to optimized profile with score {best_score:.2f}")
            self.last_optimization = time.time()
            return best_variant
        else:
            logger.info("Current profile remains optimal")
            return current_profile

    def _profile_hash(self, profile: ConnectionProfile) -> str:
        profile_dict = {
            "fragmentation_size": profile.fragmentation_size,
            "delay_jitter": profile.delay_jitter_ms,
            "burst_size": profile.packet_burst_size,
            "record_split": profile.tls_record_split_size,
            "grease": profile.enable_grease,
            "sni": profile.custom_sni,
            "padding": profile.padding_strategy,
            "modes": [mode.value for mode in profile.obfuscation_modes]
        }
        return hashlib.md5(json.dumps(profile_dict, sort_keys=True).encode()).hexdigest()

class TLSSessionObfuscator:
    """Advanced TLS session obfuscation with multiple evasion techniques"""

    def __init__(self, profile: ConnectionProfile):
        self.profile = profile
        self.grease_values = [
            0x0A0A, 0x1A1A, 0x2A2A, 0x3A3A, 0x4A4A, 0x5A5A, 0x6A6A, 0x7A7A,
            0x8A8A, 0x9A9A, 0xAAAA, 0xBABA, 0xCACA, 0xDADA, 0xEAEA, 0xFAFA
        ]

    def create_obfuscated_context(self) -> ssl.SSLContext:
        context = ssl.create_default_context()
        context.set_ciphers(self._get_evasive_cipher_suite())
        
        # TLS version obfuscation
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3

        # Add GREASE cipher suites if enabled
        if self.profile.enable_grease:
            self._inject_grease_ciphers(context)

        # Custom SNI
        if self.profile.custom_sni:
            context.set_servername_callback(self._custom_sni_callback)

        return context

    def _get_evasive_cipher_suite(self) -> str:
        # Prioritize less commonly blocked cipher suites
        evasive_ciphers = [
            'ECDHE-ECDSA-AES128-GCM-SHA256',
            'ECDHE-RSA-AES128-GCM-SHA256', 
            'ECDHE-ECDSA-CHACHA20-POLY1305',
            'ECDHE-RSA-CHACHA20-POLY1305',
            'ECDHE-ECDSA-AES256-GCM-SHA384',
            'ECDHE-RSA-AES256-GCM-SHA384',
            'AES128-GCM-SHA256',
            'AES256-GCM-SHA384'
        ]
        return ":".join(evasive_ciphers)

    def _inject_grease_ciphers(self, context: ssl.SSLContext):
        # This is a simplified implementation - actual GREASE injection requires
        # modifying the TLS ClientHello at the packet level
        pass  # Implementation would require custom TLS extension handling

    def _custom_sni_callback(self, ssl_obj, hostname, context):
        # Set the actual target after connection
        ssl_obj.set_servername(self.profile.custom_sni)

    def fragment_tls_records(self, data: bytes) -> List[bytes]:
        """Split TLS records into smaller fragments to evade DPI pattern matching"""
        if ObfuscationMode.TLS_FRAGMENTATION not in self.profile.obfuscation_modes:
            return [data]

        fragments = []
        max_fragment_size = self.profile.fragmentation_size

        for i in range(0, len(data), max_fragment_size):
            fragment = data[i:i + max_fragment_size]
            fragments.append(fragment)

        return fragments

    def add_tls_padding(self, data: bytes) -> bytes:
        """Add padding to TLS records to change their signature"""
        if ObfuscationMode.TLS_PADDING not in self.profile.obfuscation_modes:
            return data

        if self.profile.padding_strategy == "random":
            padding_length = random.randint(1, 255)
        elif self.profile.padding_strategy == "fixed":
            padding_length = 128
        else:
            return data

        # Add padding (simplified - actual TLS padding is more complex)
        padding = bytes([padding_length] * padding_length)
        return data + padding

class TCPObfuscator:
    """TCP-level obfuscation techniques"""

    def __init__(self, profile: ConnectionProfile):
        self.profile = profile

    async def send_with_obfuscation(self, writer: asyncio.StreamWriter, data: bytes):
        """Send data with TCP-level obfuscation techniques"""
        
        # Fragment data at TCP level
        if ObfuscationMode.TCP_SEGMENTATION in self.profile.obfuscation_modes:
            await self._send_with_segmentation(writer, data)
        else:
            await self._send_with_jitter(writer, data)

    async def _send_with_segmentation(self, writer: asyncio.StreamWriter, data: bytes):
        """Send data in small bursts with jitter between packets"""
        segment_size = min(self.profile.fragmentation_size, 1460)  # Typical MTU
        burst_size = self.profile.packet_burst_size
        min_delay, max_delay = self.profile.delay_jitter_ms

        for i in range(0, len(data), segment_size * burst_size):
            burst = data[i:i + segment_size * burst_size]
            
            # Send burst of packets
            for j in range(0, len(burst), segment_size):
                segment = burst[j:j + segment_size]
                writer.write(segment)
                await writer.drain()
                
                # Small delay within burst (micro-jitter)
                if j + segment_size < len(burst):
                    await asyncio.sleep(random.uniform(0.001, 0.005))
            
            # Larger delay between bursts
            if i + segment_size * burst_size < len(data):
                jitter_delay = random.uniform(min_delay / 1000.0, max_delay / 1000.0)
                await asyncio.sleep(jitter_delay)

    async def _send_with_jitter(self, writer: asyncio.StreamWriter, data: bytes):
        """Send with randomized timing"""
        min_delay, max_delay = self.profile.delay_jitter_ms
        writer.write(data)
        await writer.drain()
        
        # Add jitter after sending
        jitter_delay = random.uniform(min_delay / 1000.0, max_delay / 1000.0)
        await asyncio.sleep(jitter_delay)

class RedditDPIBypasser:
    """Main class for bypassing Reddit DPI blocks"""

    def __init__(self, target_host: str = "www.reddit.com", target_port: int = 443):
        self.target_host = target_host
        self.target_port = target_port
        self.optimizer = SmartParameterOptimizer()
        self.current_profile = ConnectionProfile()
        self.session_count = 0
        self.successful_sessions = 0

    async def create_obfuscated_connection(self) -> Optional[Tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
        """Create an obfuscated connection to the target"""
        try:
            # Optimize profile if needed
            if self.optimizer.should_optimize():
                self.current_profile = self.optimizer.generate_optimized_profile(self.current_profile)

            # Create SSL context with obfuscation
            tls_obfuscator = TLSSessionObfuscator(self.current_profile)
            ssl_context = tls_obfuscator.create_obfuscated_context()

            # Connect with TCP obfuscation
            tcp_obfuscator = TCPObfuscator(self.current_profile)

            # Resolve target IP
            addr_info = await asyncio.get_event_loop().getaddrinfo(
                self.target_host, self.target_port, 
                family=socket.AF_INET, type=socket.SOCK_STREAM
            )
            target_ip = addr_info[0][4][0]

            # Create raw connection
            reader, writer = await asyncio.open_connection(target_ip, self.target_port, ssl=None)
            
            # Perform TLS handshake with obfuscation
            await self._perform_obfuscated_handshake(reader, writer, ssl_context, tls_obfuscator, tcp_obfuscator)
            
            self.session_count += 1
            return reader, writer

        except Exception as e:
            logger.error(f"Failed to create obfuscated connection: {e}")
            return None

    async def _perform_obfuscated_handshake(self, reader: asyncio.StreamReader, 
                                          writer: asyncio.StreamWriter, 
                                          ssl_context: ssl.SSLContext,
                                          tls_obfuscator: TLSSessionObfuscator,
                                          tcp_obfuscator: TCPObfuscator):
        """Perform TLS handshake with advanced obfuscation techniques"""
        
        # Create SSL socket wrapper
        sock = writer.get_extra_info('socket')
        ssl_sock = ssl_context.wrap_socket(
            sock, 
            server_hostname=self.target_host if not self.current_profile.custom_sni else self.current_profile.custom_sni,
            do_handshake_on_connect=False
        )

        # Extract the raw ClientHello
        client_hello = self._extract_client_hello(ssl_sock)
        
        if client_hello:
            # Apply TLS record splitting
            if ObfuscationMode.TLS_RECORD_SPLITTING in self.current_profile.obfuscation_modes:
                client_hello = self._split_tls_records(client_hello, self.current_profile.tls_record_split_size)
            
            # Apply TLS fragmentation
            fragments = tls_obfuscator.fragment_tls_records(client_hello)
            
            # Send with TCP obfuscation
            for fragment in fragments:
                # Add padding if enabled
                fragment = tls_obfuscator.add_tls_padding(fragment)
                await tcp_obfuscator.send_with_obfuscation(writer, fragment)
                
                # Small delay between fragments
                await asyncio.sleep(random.uniform(0.001, 0.01))
        
        # Complete handshake
        try:
            ssl_sock.do_handshake()
            # Replace the writer's transport with the SSL-wrapped version
            # This is a simplification - actual implementation would be more complex
        except ssl.SSLError as e:
            logger.error(f"SSL handshake failed: {e}")
            raise

    def _extract_client_hello(self, ssl_sock) -> Optional[bytes]:
        """Extract ClientHello message (simplified implementation)"""
        # In a real implementation, this would involve creating a custom SSL engine
        # that can intercept and modify the ClientHello before sending
        # This is a placeholder for the complex logic required
        return b''  # Placeholder

    def _split_tls_records(self, data: bytes, split_factor: int) -> bytes:
        """Split TLS records into multiple smaller records"""
        if len(data) < 5:  # Minimum TLS record size
            return data

        # Parse TLS record header
        record_type = data[0]
        protocol_version = data[1:3]
        record_length = struct.unpack('!H', data[3:5])[0]

        if len(data) < 5 + record_length:
            return data  # Incomplete record

        payload = data[5:5 + record_length]
        remaining = data[5 + record_length:]

        # Split payload
        chunk_size = max(1, len(payload) // split_factor)
        chunks = [payload[i:i + chunk_size] for i in range(0, len(payload), chunk_size)]

        # Rebuild records
        result = b''
        for chunk in chunks:
            chunk_record = struct.pack('!B', record_type) + protocol_version + struct.pack('!H', len(chunk)) + chunk
            result += chunk_record

        return result + remaining

    async def make_request(self, path: str = "/", headers: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Make an HTTP request through the obfuscated connection"""
        start_time = time.time()
        profile_hash = hashlib.md5(str(self.current_profile.__dict__).encode()).hexdigest()

        try:
            connection = await self.create_obfuscated_connection()
            if not connection:
                self.optimizer.record_attempt(profile_hash, False, time.time() - start_time)
                return None

            reader, writer = connection

            # Build HTTP request
            if headers is None:
                headers = {
                    "User-Agent": self._get_random_user_agent(),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Cache-Control": "max-age=0"
                }

            request_lines = [f"GET {path} HTTP/1.1", f"Host: {self.target_host}"]
            for key, value in headers.items():
                request_lines.append(f"{key}: {value}")
            request_lines.append("")  # Empty line to end headers
            request_lines.append("")  # Empty body

            request = "\r\n".join(request_lines).encode()

            # Send request with obfuscation
            tcp_obfuscator = TCPObfuscator(self.current_profile)
            await tcp_obfuscator.send_with_obfuscation(writer, request)

            # Read response
            response = await self._read_response(reader)

            latency = time.time() - start_time
            self.optimizer.record_attempt(profile_hash, True, latency)
            self.successful_sessions += 1

            writer.close()
            await writer.wait_closed()

            return response

        except Exception as e:
            latency = time.time() - start_time
            self.optimizer.record_attempt(profile_hash, False, latency)
            logger.error(f"Request failed: {e}")
            return None

    async def _read_response(self, reader: asyncio.StreamReader) -> str:
        """Read HTTP response with timeout"""
        try:
            response_data = b""
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=10.0)
                if not chunk:
                    break
                response_data += chunk
                
                # Simple check for end of headers (incomplete implementation)
                if b"\r\n\r\n" in response_data:
                    # Look for Content-Length or chunked encoding
                    # This is simplified - real implementation would parse HTTP properly
                    break

            return response_data.decode('utf-8', errors='ignore')
        except asyncio.TimeoutError:
            raise Exception("Response timeout")

    def _get_random_user_agent(self) -> str:
        """Return a random user agent to avoid fingerprinting"""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
        ]
        return random.choice(user_agents)

    def get_statistics(self) -> Dict[str, Any]:
        """Return connection statistics"""
        success_rate = self.successful_sessions / max(self.session_count, 1)
        return {
            "total_sessions": self.session_count,
            "successful_sessions": self.successful_sessions,
            "success_rate": success_rate,
            "current_profile": self.current_profile.__dict__
        }

async def main():
    parser = argparse.ArgumentParser(description="Advanced Reddit DPI Circumvention Tool")
    parser.add_argument("--host", default="www.reddit.com", help="Target host")
    parser.add_argument("--port", type=int, default=443, help="Target port")
    parser.add_argument("--path", default="/", help="Request path")
    parser.add_argument("--test", action="store_true", help="Run connectivity test")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()

    bypasser = RedditDPIBypasser(args.host, args.port)

    if args.test:
        logger.info("Testing connection to Reddit...")
        response = await bypasser.make_request(args.path)
        if response:
            logger.info("Connection successful!")
            print(f"Response preview: {response[:500]}...")
        else:
            logger.error("Connection failed")
            sys.exit(1)
    elif args.stats:
        # Make a few test requests first
        for i in range(3):
            logger.info(f"Test request {i+1}/3")
            await bypasser.make_request("/")
            await asyncio.sleep(1)
        
        stats = bypasser.get_statistics()
        print(json.dumps(stats, indent=2))
    else:
        # Interactive mode
        print("Reddit DPI Bypasser - Type 'quit' to exit")
        while True:
            try:
                path = input("Enter path to request (default: /): ").strip()
                if path.lower() == 'quit':
                    break
                if not path:
                    path = "/"
                
                logger.info(f"Requesting {path}...")
                response = await bypasser.make_request(path)
                
                if response:
                    print(f"\nResponse received ({len(response)} bytes):")
                    print("-" * 50)
                    print(response[:1000])
                    if len(response) > 1000:
                        print("... (truncated)")
                    print("-" * 50)
                else:
                    print("Request failed")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

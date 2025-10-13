from __future__ import annotations

import hashlib
import glob
import json
import logging
import os
import platform
import psutil
import re
import shlex
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
import traceback
import uuid
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, STDOUT
from typing import Any, Callable

import docker
from docker.models.containers import Container

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Port management constants
DEFAULT_PORT_RANGE_START = 10000
DEFAULT_PORT_RANGE_END = 20000
DOCKER_START_UP_DELAY = 1
DOCKER_COMPOSE_TERMINATION_DELAY = 100
DOCKER_COMPOSE_STARTUP_DELAY = 600


##### IP and Port Management Functions #####

def get_vmnet1_ip():
    interfaces = psutil.net_if_addrs()
    system = platform.system()

    target_names = []
    if system == "Windows":
        # Windows adapter names often include "VMware Network Adapter VMnet1"
        target_names = [name for name in interfaces if "VMnet1" in name or "VMware" in name]
    else:
        # On Linux/macOS the interface might be called 'vmnet1'
        target_names = [name for name in interfaces if "vmnet1" in name.lower()]

    for iface in target_names:
        for addr in interfaces[iface]:
            if addr.family == socket.AF_INET:
                return addr.address
    return None


def is_port_in_use(port: int, host: str = 'localhost') -> bool:
    """Check if a port is currently in use"""
    import socket
    
    # Check if we can bind to the port (TCP)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return False  # Port is available
    except OSError:
        pass  # Port might be in use, check further
    
    # Also check if anything is listening on the port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.1)  # Very short timeout
            result = sock.connect_ex((host, port))
            return result == 0  # If connection succeeds, port is in use
    except:
        pass
    
    return True  # Assume in use if we can't determine


def get_available_port(start_port: int = DEFAULT_PORT_RANGE_START, end_port: int = DEFAULT_PORT_RANGE_END, host: str = 'localhost') -> int:
    """Find an available port in the specified range by checking actual port usage"""
    import random
    
    # Create a randomized list of ports to try to avoid patterns
    port_range = list(range(start_port, end_port + 1))
    random.shuffle(port_range)
    
    for port in port_range:
        if not is_port_in_use(port, host):
            logger.debug(f"Found available port: {port}")
            return port
    
    raise RuntimeError(f"No available ports found in range {start_port}-{end_port}")


##### End of IP and Port Management Functions #####


##### Docker Compose Management Functions #####

def get_docker_compose(
    docker_compose_path: Path, 
    container_name_suffix: str | None = None,
    challenge_internal_port: int | None = None
) -> tuple[Path, dict[str, int]]:
    """
    Start docker-compose services with optional dynamic port allocation.
    
    Args:
        docker_compose_path: Path to the docker-compose.yml file
        container_name_suffix: Optional suffix for container names to avoid conflicts
        challenge_internal_port: Optional internal port from challenge.json that should be exposed
        
    Returns:
        Tuple of (compose_path, port_mappings) where port_mappings maps internal ports to external ports
    """
    actual_compose_path = docker_compose_path
    port_mappings = {}
    
    if container_name_suffix:
        # Generate unique network name for this instance
        dynamic_network_name = f"ctfnet-{container_name_suffix}"
        
        # Get available ports for the services
        import yaml
        try:
            with open(docker_compose_path) as f:
                compose_data = yaml.safe_load(f)
            
            # Collect all internal ports that need mapping
            if "services" in compose_data:
                for service_config in compose_data["services"].values():
                    if "ports" in service_config:
                        for port_mapping in service_config["ports"]:
                            if isinstance(port_mapping, str) and ":" in port_mapping:
                                external_port, internal_port = port_mapping.split(":", 1)
                                try:
                                    available_port = get_available_port()
                                    port_mappings[internal_port] = available_port
                                    logger.debug(f"Mapped internal port {internal_port} to external port {available_port}")
                                except RuntimeError as e:
                                    logger.warning(f"Could not allocate dynamic port for {internal_port}: {e}")
                            elif isinstance(port_mapping, int):
                                # Handle integer port (just internal port specified)
                                internal_port = str(port_mapping)
                                try:
                                    available_port = get_available_port()
                                    port_mappings[internal_port] = available_port
                                    logger.debug(f"Mapped internal port {internal_port} to external port {available_port}")
                                except RuntimeError as e:
                                    logger.warning(f"Could not allocate dynamic port for {internal_port}: {e}")
            
            # Handle challenge internal port if specified and not already mapped
            if challenge_internal_port is not None:
                internal_port_str = str(challenge_internal_port)
                if internal_port_str not in port_mappings:
                    try:
                        available_port = get_available_port()
                        port_mappings[internal_port_str] = available_port
                        logger.debug(f"Mapped challenge internal port {internal_port_str} to external port {available_port}")
                    except RuntimeError as e:
                        logger.warning(f"Could not allocate dynamic port for challenge internal port {internal_port_str}: {e}")
                        
        except Exception as e:
            logger.warning(f"Failed to parse compose file for dynamic ports: {e}")
        
        if port_mappings:
            # Create modified docker-compose file
            try:
                actual_compose_path = create_dynamic_docker_compose(
                    docker_compose_path, 
                    container_name_suffix,
                    dynamic_network_name,
                    port_mappings
                )
                logger.info(f"Created dynamic docker-compose at {actual_compose_path} with port mappings: {port_mappings}")
            except Exception as e:
                logger.error(f"Failed to create dynamic docker-compose: {e}")
                actual_compose_path = docker_compose_path
                port_mappings = {}
    
    startup_cmd = [
        "docker",
        "compose",
        "-f",
        str(actual_compose_path),
        "up",
        "-d",
        "--force-recreate",
    ]
    logger.debug("Starting docker-compose with command: %s", shlex.join(startup_cmd))
    compose = subprocess.Popen(
        startup_cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
        bufsize=1,  # line buffered
    )
    stdout, stderr = compose.communicate(timeout=DOCKER_COMPOSE_STARTUP_DELAY)
    if compose.returncode != 0:
        logger.error(f"Compose failed (code {compose.returncode}):\nSTDOUT: {stdout}\nSTDERR: {stderr}")
    else:
        # print(stdout)  # success info from stdout
        if stderr.strip():
            logger.info(f"Compose stderr (non-critical): {stderr}")
    
    return actual_compose_path, port_mappings


def create_dynamic_docker_compose(
    original_compose_path: Path, 
    container_name_suffix: str,
    dynamic_network_name: str,
    port_mappings: dict[str, int] | None = None
) -> Path:
    """
    Create a modified docker-compose file with dynamic port mappings and network names.
    
    Args:
        original_compose_path: Path to the original docker-compose.yml
        container_name_suffix: Unique suffix to append to container names
        dynamic_network_name: Unique network name for this instance
        port_mappings: Optional dict mapping original internal ports to new external ports
    
    Returns:
        Path to the temporary modified docker-compose file
    """
    import yaml
    
    with open(original_compose_path) as f:
        compose_data = yaml.safe_load(f)
    
    # Modify service names and ports
    if "services" in compose_data:
        new_services = {}
        for service_name, service_config in compose_data["services"].items():
            # Add suffix to service name
            new_service_name = f"{service_name}-{container_name_suffix}"
            new_service_config = service_config.copy()
            
            # Handle port mappings
            if "ports" in new_service_config:
                new_ports = []
                for port_mapping in new_service_config["ports"]:
                    if isinstance(port_mapping, str) and ":" in port_mapping:
                        external_port, internal_port = port_mapping.split(":", 1)
                        # Use mapped port if available, otherwise find a new one
                        if port_mappings and internal_port in port_mappings:
                            new_ports.append(f"{port_mappings[internal_port]}:{internal_port}")
                        else:
                            # Try to find an available port for unmapped ports
                            try:
                                available_port = get_available_port()
                                new_ports.append(f"{available_port}:{internal_port}")
                                if port_mappings is not None:
                                    port_mappings[internal_port] = available_port
                                logger.debug(f"Auto-assigned port {available_port} for internal port {internal_port}")
                            except RuntimeError:
                                logger.warning(f"Could not find available port for {port_mapping}, keeping original")
                                new_ports.append(port_mapping)
                    elif isinstance(port_mapping, int):
                        # Handle integer port (just internal port specified)
                        internal_port = str(port_mapping)
                        if port_mappings and internal_port in port_mappings:
                            new_ports.append(f"{port_mappings[internal_port]}:{internal_port}")
                        else:
                            try:
                                available_port = get_available_port()
                                new_ports.append(f"{available_port}:{internal_port}")
                                if port_mappings is not None:
                                    port_mappings[internal_port] = available_port
                                logger.debug(f"Auto-assigned port {available_port} for internal port {internal_port}")
                            except RuntimeError:
                                logger.warning(f"Could not find available port for {port_mapping}, keeping original")
                                new_ports.append(port_mapping)
                    else:
                        new_ports.append(port_mapping)
                new_service_config["ports"] = new_ports
            elif port_mappings:
                # No explicit ports in compose file, but we have port mappings to add
                # This handles cases where the compose file doesn't specify ports but challenge.json does
                new_ports = []
                for internal_port, external_port in port_mappings.items():
                    new_ports.append(f"{external_port}:{internal_port}")
                    logger.debug(f"Adding port mapping {external_port}:{internal_port} to service {new_service_name}")
                if new_ports:
                    new_service_config["ports"] = new_ports
            
            # Update network references
            if "networks" in new_service_config:
                if isinstance(new_service_config["networks"], list):
                    # Simple list format
                    new_networks = []
                    for net in new_service_config["networks"]:
                        if net == "ctfnet":
                            new_networks.append(dynamic_network_name)
                        else:
                            new_networks.append(net)
                    new_service_config["networks"] = new_networks
                elif isinstance(new_service_config["networks"], dict):
                    # Dict format with aliases
                    new_networks = {}
                    for net_name, net_config in new_service_config["networks"].items():
                        if net_name == "ctfnet":
                            new_networks[dynamic_network_name] = net_config
                        else:
                            new_networks[net_name] = net_config
                    new_service_config["networks"] = new_networks
            
            new_services[new_service_name] = new_service_config
        
        compose_data["services"] = new_services
    
    # Update network definitions
    # TODO (kaixin): Check if the network configuration is correct here
    if "networks" in compose_data:
        new_networks = {}
        for net_name, net_config in compose_data["networks"].items():
            if net_name == "ctfnet":
                # Create a new internal network instead of external
                new_networks[dynamic_network_name] = {
                    "driver": "bridge",
                    "name": dynamic_network_name
                }
            else:
                new_networks[net_name] = net_config
        compose_data["networks"] = new_networks
    
    # Copy the whole folder to a temporary folder, and write the new compose file there

    challenge_folder = original_compose_path.parent
    temp_dir = Path(tempfile.mkdtemp(prefix=f"ctf-docker-compose-{container_name_suffix}-"))
    temp_compose_path = Path(temp_dir) / "docker-compose.yml"
    # Copy the folder contents to the temporary directory
    
    # Create subfolder inside temp dir for the challenge
    dest_challenge_dir = temp_dir / challenge_folder.name
    shutil.copytree(challenge_folder, dest_challenge_dir)

    # Overwrite the docker-compose.yml inside the copied folder
    temp_compose_path = dest_challenge_dir / "docker-compose.yml"
    with open(temp_compose_path, 'w') as f:
        yaml.dump(compose_data, f, default_flow_style=False)

    return temp_compose_path


def terminate_docker_compose(docker_compose_path: Path) -> None:
    terminate_cmd = [
        "docker",
        "compose",
        "-f",
        str(docker_compose_path),
        "down",
    ]
    logger.debug("Terminating docker-compose with command: %s", shlex.join(terminate_cmd))
    compose = subprocess.Popen(
        terminate_cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
        bufsize=1,  # line buffered
    )
    _, error = compose.communicate(timeout=DOCKER_COMPOSE_TERMINATION_DELAY)
    if error:
        logger.error(f"Unexpected compose termination error: {error}")


def attach_network_interface_to_container(container_name: str, network_name: str = "ctfnet") -> None:
    """
    Attach a network interface to a container.
    
    Args:
        container_name: Name of the container to attach network to
        network_name: Name of the network to attach (defaults to 'ctfnet')
    """
    # First ensure the network exists
    client = docker.from_env()
    try:
        client.networks.get(network_name)
    except docker.errors.NotFound:
        # Create the network if it doesn't exist
        try:
            client.networks.create(network_name, driver="bridge")
            logger.debug(f"Created network {network_name}")
        except docker.errors.APIError as e:
            logger.warning(f"Failed to create network {network_name}: {e}")
    
    cmd = [
        "docker",
        "network",
        "connect",
        network_name,
        container_name,
    ]
    logger.debug("Attaching NIC to container with command: %s", shlex.join(cmd))
    compose = subprocess.Popen(
        cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
        bufsize=1,  # line buffered
    )
    _, error = compose.communicate(timeout=DOCKER_START_UP_DELAY)
    if error:
        logger.error(f"Unexpected compose setup error: {error}")
        raise RuntimeError(error)

##### End of Docker Compose Management Functions #####


##### Cleanup Functions #####

def cleanup_dynamic_network(network_name: str) -> None:
    """
    Clean up a specific dynamic CTF network.
    
    Args:
        network_name: Name of the network to remove (e.g., 'ctfnet-abc123')
    """
    if not network_name or network_name == "ctfnet":
        # Don't remove the base ctfnet network
        return
    
    try:
        client = docker.from_env()
        network = client.networks.get(network_name)
        network.remove()
        logger.debug(f"Cleaned up dynamic network: {network_name}")
    except docker.errors.NotFound:
        logger.debug(f"Dynamic network {network_name} not found, likely already removed")
    except docker.errors.APIError as e:
        logger.warning(f"Failed to remove dynamic network {network_name}: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error removing dynamic network {network_name}: {e}")


def cleanup_all_dynamic_networks() -> None:
    """
    Comprehensive cleanup of ALL dynamic CTF networks.
    This function finds and removes all networks matching the 'ctfnet-*' pattern,
    similar to the external cleanup script approach.
    """
    try:
        client = docker.from_env()
        networks = client.networks.list()
        
        # Find all dynamic ctfnet networks (those starting with 'ctfnet-')
        dynamic_networks = [net for net in networks if net.name.startswith('ctfnet-')]
        
        if dynamic_networks:
            logger.debug(f"Found {len(dynamic_networks)} dynamic CTF networks to clean up")
            for network in dynamic_networks:
                try:
                    # First try to remove directly
                    network.remove()
                    logger.debug(f"Cleaned up dynamic network: {network.name}")
                except docker.errors.APIError as e:
                    if "has active endpoints" in str(e):
                        # Network has active containers, try to disconnect them first
                        logger.debug(f"Network {network.name} has active endpoints, disconnecting containers...")
                        try:
                            # Reload network to get fresh endpoint info
                            network.reload()
                            # Disconnect all containers from this network
                            for container_id, endpoint_config in network.attrs.get('Containers', {}).items():
                                try:
                                    container = client.containers.get(container_id)
                                    network.disconnect(container, force=True)
                                    logger.debug(f"Disconnected container {container.name} from network {network.name}")
                                except Exception as disconnect_e:
                                    logger.debug(f"Failed to disconnect container {container_id}: {disconnect_e}")
                            
                            # Now try to remove the network again
                            network.remove()
                            logger.debug(f"Cleaned up dynamic network after disconnecting containers: {network.name}")
                        except Exception as cleanup_e:
                            logger.warning(f"Failed to forcefully clean up network {network.name}: {cleanup_e}")
                    else:
                        logger.warning(f"Failed to remove dynamic network {network.name}: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error removing dynamic network {network.name}: {e}")
        else:
            logger.debug("No dynamic CTF networks found to clean up")
    
    except docker.errors.DockerException as e:
        logger.warning(f"Docker error during network cleanup: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error during comprehensive network cleanup: {e}")


def cleanup_dynamic_resources() -> None:
    """
    Comprehensive cleanup of dynamic CTF resources including networks and temporary files.
    This function provides thorough cleanup similar to the external cleanup script.
    """
    # Clean up all dynamic networks
    cleanup_all_dynamic_networks()
    
    # Clean up temporary docker-compose files
    try:
        temp_files = glob.glob('/tmp/ctf-docker-compose-*')
        for temp_file in temp_files:
            path = Path(temp_file)
            try:
                if path.is_file():
                    path.unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
                elif path.is_dir():
                    shutil.rmtree(path)
                    logger.debug(f"Cleaned up temporary directory: {temp_file}")
            except FileNotFoundError:
                pass  # Already removed
            except Exception as e:
                logger.warning(f"Failed to remove temporary file/directory {temp_file}: {e}")
        if temp_files:
            logger.debug(f"Cleaned up {len(temp_files)} temporary docker-compose files/directories")
    except Exception as e:
        logger.warning(f"Error during temporary file cleanup: {e}")


def force_cleanup_all_ctf_resources() -> dict[str, int]:
    """
    Force cleanup of ALL CTF-related resources. 
    This is a comprehensive cleanup function that can be used for manual cleanup
    or in cleanup scripts. It mimics the behavior of the external cleanup script.
    
    Returns:
        Dictionary with counts of cleaned up resources
    """
    cleanup_stats = {
        "networks_removed": 0,
        "temp_files_removed": 0,
        "errors": 0
    }
    
    try:
        client = docker.from_env()
        
        # Find and remove all CTF networks (ctfnet-* and ctfnet)
        networks = client.networks.list()
        ctf_networks = [net for net in networks if net.name.startswith('ctfnet')]
        
        for network in ctf_networks:
            try:
                # Try to remove the network
                network.remove()
                cleanup_stats["networks_removed"] += 1
                logger.info(f"Removed CTF network: {network.name}")
            except docker.errors.APIError as e:
                if "has active endpoints" in str(e):
                    logger.warning(f"Network {network.name} has active containers, skipping")
                elif "not found" in str(e).lower():
                    logger.debug(f"Network {network.name} already removed")
                else:
                    logger.warning(f"Failed to remove network {network.name}: {e}")
                    cleanup_stats["errors"] += 1
            except Exception as e:
                logger.warning(f"Unexpected error removing network {network.name}: {e}")
                cleanup_stats["errors"] += 1
    
    except docker.errors.DockerException as e:
        logger.error(f"Docker error during comprehensive cleanup: {e}")
        cleanup_stats["errors"] += 1
    except Exception as e:
        logger.error(f"Unexpected error during comprehensive cleanup: {e}")
        cleanup_stats["errors"] += 1
    
    # Clean up temporary files
    try:
        import glob
        temp_files = glob.glob('/tmp/ctf-docker-compose-*')
        for temp_file in temp_files:
            try:
                Path(temp_file).unlink()
                cleanup_stats["temp_files_removed"] += 1
                logger.debug(f"Removed temporary file: {temp_file}")
            except FileNotFoundError:
                pass  # Already removed
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
                cleanup_stats["errors"] += 1
    except Exception as e:
        logger.warning(f"Error during temporary file cleanup: {e}")
        cleanup_stats["errors"] += 1
    
    return cleanup_stats


##### CTF Environment Helper Functions #####

def extract_flag_format(flag: str) -> str:
    flag_format = re.sub(r"{.*}$", "{...}", flag)
    return flag_format if flag_format != flag else "..."


##### End of CTF Environment Helper Functions #####


@dataclass(frozen=True)
class CTFEnvironmentArguments:
    """Configure data sources and setup instructions for the environment in which we solve the tasks."""

    # Source of issue statement/problem statement. To run over a batch of issues: Path to a data file
    # (`json`, `jsonl`) or directory. To run over single issue: github issue url or path to markdown file
    # with problem statement or problem statement as text prefixed with `text://`.
    data_path: str
    # Enable environment logger.
    verbose: bool = False

    def __post_init__(self):
        # Check args here
        pass


class CTFWebServerManager:
    def __init__(self, args: CTFEnvironmentArguments):
        super().__init__()
        self.args = args
        
        self.logger = logger
        if not self.args.verbose:
            # fixme: This creates problems if we have multiple instances of this class
            self.logger.disabled = True

        # short uuid
        self.env_id = uuid.uuid4().hex[:6]
        # Load Task Instances
        self.data_path = self.args.data_path

        self.challenge: dict[str, Any] | None = None
        self._load_challenge(str(Path(self.data_path) / "challenge.json"))
        self.docker_compose: Path | None = None
        
        # Dynamic port allocation for CTF challenges
        self.port_mappings: dict[str, int] = {}
        self.dynamic_network_name: str | None = None

    def _load_challenge(self, config_path: str | None = None) -> None:
        """For CTF challenges"""
        challenge = json.loads(Path(config_path).read_text())
        self.challenge = challenge.copy()
        self.challenge["files"] = challenge.get("files", [])
        self.challenge["points"] = challenge.get("points", 10)
        
        if (Path(config_path).parent / "docker-compose.yml").is_file():
            logger.debug(f"Found docker_compose file in {Path(config_path).parent}")
            self.challenge["docker_compose"] = Path(config_path).parent / "docker-compose.yml"
        self.challenge["port"] = challenge.get("internal_port") or challenge.get("port")
        if "box" in challenge:
            self.challenge["server_name"] = challenge["box"] or "127.0.0.1"
        else:
            self.challenge["server_name"] = ""
        self.challenge["file_path"] = config_path

        # Set server description for the challenge
        server_name = self.challenge["server_name"]
        port = self.challenge["port"]
        if server_name is None or port is None:
            self.challenge["server_description"] = ""
        else:
            self.challenge["server_description"] = (
                f"The challenge web server is running on `{server_name}` port `{port}`. You can access it via the browser: `http://{server_name}:{port}`."
            )
        # Set problem statement for the challenge
        self.challenge["problem_statement"] = f"{self.challenge['name']} {self.challenge['description']}"
        self.logger.info(f"Loaded challenge: {self.challenge['name']}")


    def _init_docker_compose(self) -> None:
        """
        Handles docker compose initialization for challenge with docker compose file.
        """
        if self.challenge is not None and self.challenge.get("docker_compose") is not None:
            self.logger.info("🌱 Initializing docker compose for challenge")
            # Generate unique suffix for this instance to avoid conflicts
            # Use env id as unique suffix
            container_suffix = self.env_id
            self.dynamic_network_name = f"ctfnet-{container_suffix}"
            
            self.docker_compose, self.port_mappings = get_docker_compose(
                self.challenge["docker_compose"],
                container_name_suffix=container_suffix,
                challenge_internal_port=self.challenge.get("port")
            )
            self.logger.info("🌱 Initialized docker compose for challenge")
            if self.port_mappings:
                self.logger.info(f"🔌 Dynamic port mappings: {self.port_mappings}")
                self._update_server_description_with_port_mapping(self.port_mappings)

    def _update_server_description_with_port_mapping(self, port_mappings: dict[str, int]) -> None:
        """Update server description after dynamic port mapping is established
        
        Args:
            port_mappings: Dictionary mapping internal ports (as strings) to external ports
        """
            
        challenge = self.challenge
        internal_port = challenge.get("port")
        # server_name = challenge.get("server_name")

        # Get the VMnet1 IP address if available
        server_name = get_vmnet1_ip()
        if not server_name:
            raise RuntimeError("Could not determine VMnet1 IP address. Ensure VMware is properly installed.")
        
        
        if internal_port is not None and str(internal_port) in port_mappings:
            external_port = port_mappings[str(internal_port)]
            # Update the server description with the external port
            # IMPORTANT: For container-to-container communication, ALWAYS use internal port and service name
            # External ports are only for host-to-container communication
            # Since the SWE agent runs inside a container, it should use the service name and internal port

            # (Kaixin): In CTFWorld, agent runs on the host (or host's vm). We need to provide the external port
            self.challenge["server_description"] = (
                f"The challenge web server is running on `{server_name}` port `{external_port}`. You can access it via the browser: `http://{server_name}:{external_port}`."
            )
            # Store the port mapping info for reference
            challenge["external_port"] = external_port
            challenge["port_mapping"] = port_mappings
            challenge["server_url"] = f"http://{server_name}:{external_port}"
            self.logger.info(f"Updated server url to: {challenge['server_url']}")

    def close(self) -> None:
        """
        Handle environment shutdown
        """
        self.logger.info("Beginning environment shutdown...")

        if self.docker_compose is not None:
            try:
                terminate_docker_compose(self.docker_compose)
            except KeyboardInterrupt:
                raise
            except:
                self.logger.warning("Failed to terminate docker compose", exc_info=True)
            else:
                self.logger.debug("Terminated docker compose")
            
            # Clean up temporary docker-compose file if it was dynamically created
            if str(self.docker_compose).startswith('/tmp/ctf-docker-compose-'):
                try:
                    self.docker_compose.unlink()
                    self.logger.debug(f"Cleaned up temporary docker-compose file: {self.docker_compose}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temporary docker-compose file: {e}")
        
        # Clean up dynamic network
        try:
            # Use comprehensive cleanup to remove ALL dynamic networks
            cleanup_dynamic_resources()
            self.logger.debug("Performed comprehensive cleanup of dynamic CTF resources")
        except Exception as e:
            self.logger.warning(f"Failed to perform comprehensive cleanup: {e}")
            # Fallback to single network cleanup if comprehensive cleanup fails
            if self.dynamic_network_name:
                try:
                    cleanup_dynamic_network(self.dynamic_network_name)
                    self.logger.debug(f"Fallback: cleaned up dynamic network: {self.dynamic_network_name}")
                except Exception as fallback_e:
                    self.logger.warning(f"Failed to clean up dynamic network during fallback: {fallback_e}")

    def __enter__(self) -> "CTFWebServerManager":
        """
        Context manager enter method to allow using 'with' statement
        """
        t0 = time.perf_counter()
        self._init_docker_compose()
        self.logger.debug("Environment initialization took %.2f seconds", time.perf_counter() - t0)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Context manager exit method to handle cleanup on exit
        """
        self.close()
        if exc_type is not None:
            self.logger.error("An error occurred during the CTF environment execution", exc_info=(exc_type, exc_value, traceback))


if __name__ == "__main__":
    # Example usage
    args = CTFEnvironmentArguments(data_path="/home/ubuntu/ssd0/CTFWorld/ctf_benchmark_data/web/Guess Harder", verbose=True)
    # Run your CTF challenge logic here
    with CTFWebServerManager(args) as ctf_env:
        print(ctf_env.challenge)
        time.sleep(10000)

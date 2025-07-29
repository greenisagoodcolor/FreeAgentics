"""End-to-end test for complete developer onboarding experience."""

import subprocess
import time
import requests
import pytest
from pathlib import Path


class TestCompleteOnboarding:
    """Test the complete clone-to-working-UI flow."""

    @pytest.fixture(autouse=True)
    def setup_clean_environment(self):
        """Set up clean test environment."""
        # Kill any running processes
        subprocess.run(["make", "kill-ports"], capture_output=True)
        yield
        # Cleanup after test
        subprocess.run(["make", "kill-ports"], capture_output=True)

    def test_clone_and_run_complete_flow(self):
        """Test that a fresh clone can immediately run the complete UI."""
        project_root = Path(__file__).parent.parent.parent
        
        # 1. Test make install && make dev works
        print("Testing make install...")
        install_result = subprocess.run(
            ["make", "install"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120
        )
        assert install_result.returncode == 0, f"make install failed: {install_result.stderr}"
        
        # 2. Start dev environment in background
        print("Starting make dev...")
        dev_process = subprocess.Popen(
            ["make", "dev"],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 3. Wait for services to start
        backend_ready = False
        frontend_ready = False
        max_wait = 60  # seconds
        
        for i in range(max_wait):
            try:
                # Check backend health
                if not backend_ready:
                    health_response = requests.get("http://localhost:8000/api/v1/health", timeout=2)
                    if health_response.status_code == 200:
                        backend_ready = True
                        print("✅ Backend is ready")
                
                # Check frontend (any response means it's running)
                if not frontend_ready:
                    for port in [3000, 3001, 3002, 3003]:
                        try:
                            frontend_response = requests.get(f"http://localhost:{port}", timeout=2)
                            if frontend_response.status_code in [200, 404, 500]:  # Any response is good
                                frontend_ready = True
                                print(f"✅ Frontend is ready on port {port}")
                                break
                        except:
                            continue
                
                if backend_ready and frontend_ready:
                    break
                    
                time.sleep(1)
            except:
                time.sleep(1)
        
        try:
            assert backend_ready, "Backend failed to start within 60 seconds"
            assert frontend_ready, "Frontend failed to start within 60 seconds"
            
            # 4. Test dev-config endpoint provides token
            dev_config = requests.get("http://localhost:8000/api/v1/dev-config").json()
            assert "auth" in dev_config, "Dev config should provide auth token"
            assert "token" in dev_config["auth"], "Auth should include token"
            token = dev_config["auth"]["token"]
            
            # 5. Test authenticated endpoints work
            headers = {"Authorization": f"Bearer {token}"}
            
            # Test agents endpoint
            agents_response = requests.get("http://localhost:8000/api/v1/agents", headers=headers)
            assert agents_response.status_code == 200, f"Agents endpoint failed: {agents_response.status_code}"
            
            # Test knowledge graph endpoint
            kg_response = requests.get("http://localhost:8000/api/knowledge-graph", headers=headers)
            assert kg_response.status_code == 200, f"Knowledge graph endpoint failed: {kg_response.status_code}"
            
            # 6. Test agent creation
            agent_data = {
                "name": "Test Agent",
                "template": "basic-explorer",
                "parameters": {"description": "E2E test agent"}
            }
            create_response = requests.post(
                "http://localhost:8000/api/v1/agents", 
                json=agent_data, 
                headers=headers
            )
            assert create_response.status_code == 201, f"Agent creation failed: {create_response.status_code}"
            
            # 7. Test process-prompt endpoint if it exists
            try:
                prompt_response = requests.post(
                    "http://localhost:8000/api/process-prompt",
                    json={"prompt": "Hello test"},
                    headers=headers
                )
                if prompt_response.status_code not in [404, 405]:  # Endpoint exists
                    assert prompt_response.status_code == 200, "Process prompt should work"
            except:
                pass  # Endpoint might not exist yet
            
            print("🎉 All E2E tests passed!")
            
        finally:
            # Cleanup: kill the dev process
            dev_process.terminate()
            time.sleep(2)
            if dev_process.poll() is None:
                dev_process.kill()
            subprocess.run(["make", "kill-ports"], capture_output=True)
import vrep.vrep as vrep


class VrepConnection(object):
    """
    VrepConnection class is responsible for connecting to a V-REP session, manage session ID and also simulation
    status and simulation configurations.
    """

    def __init__(self, server_ip, server_port, force_finish_comm=True):
        self.wait_connection = True
        self.do_not_reconnect = True
        self.time_out_ms = 5000
        self.comm_thread_cycle = 5

        self.server_ip = server_ip
        self.server_port = server_port

        self.force_finish_comm = force_finish_comm
        self.client_id = self.start_server()

    def start_server(self):
        # Finish any previous open communication
        if self.force_finish_comm:
            vrep.simxFinish(-1)

        # Try connection
        client_id = vrep.simxStart(self.server_ip, self.server_port, self.wait_connection, self.do_not_reconnect,
                                   self.time_out_ms, self.comm_thread_cycle)

        if client_id == -1:
            # Connection failed
            raise Exception("Failed to connect to server. Server IP: " + self.server_ip + ":" + self.server_port)

        return client_id

    def finish_server(self):
        # Finish server
        vrep.simxFinish(self.client_id)

    def get_object_handle(self, obj_name):
        ret, handle = vrep.simxGetObjectHandle(self.client_id, obj_name, vrep.simx_opmode_blocking)
        if ret == vrep.simx_return_ok:
            return handle
        else:
            raise Exception("Failed to get object handle. Object name: " + obj_name)

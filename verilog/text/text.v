module top_module(
    input clk,
    input in,
    input reset,    // Synchronous reset
    output done
); 

parameter start_bit = 2'd2;
parameter stop_bit  = 2'd0;
parameter data_bit  = 2'd1;

reg [3:0] cnt               ;
reg [1:0] state, next_state ;
reg        flag              ;      

always @(*) begin
    case (state)
        start_bit: next_state <= (~in)?(data_bit):(start_bit);
        data_bit : next_state <= (flag)?(stop_bit):(data_bit);
        stop_bit : next_state <= (in)?(start_bit):(stop_bit);       
        default : next_state  <= start_bit;
    endcase
end
                               
always @(posedge clk ) begin
    if (reset) begin
        state <= start_bit;
    end else begin
        state <= next_state;
    end
end

always @(posedge clk ) begin
    if(reset)begin
        cnt  <= 4'd0;
        flag <= 1'b0;
    end
    if (state == data_bit) begin
        cnt <= cnt+1;    
        if(cnt == 4'd8)begin
            cnt <= 4'b0;
            flag <= 1'b1;
    end
    end else begin
        cnt <= 4'd0;
        flag <= 1'b0;
    end
end

always @(posedge clk) begin
    if(reset) begin
        done <= 1'b0;
    end else if(state == stop_bit)begin
        done <= 1'b1;
    end else begin
        done <= 1'b0;
    end
    
end


endmodule

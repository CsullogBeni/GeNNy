error\{
 NoError , PacketTooShort , NoMatch , StackOutOfBounds , HeaderTooShort , ParserTimeout , ParserInvalidArgument\}
 extern packet_in\{
 void extract \< T \> ( out T hdr );
 void extract \< T \> ( out T variableSizeHeader , in bit \< 32 \> variableFieldSizeInBits );
 T lookahead \< T \> ( );
 void advance ( in bit \< 32 \> sizeInBits );
 bit \< 32 \> length ( );
\}
 extern packet_out\{
 void emit \< T \> ( in T hdr );
\}
 extern void verify ( in bool check , in error toSignal );
 @ noWarn ( \"unused\" ) action NoAction ( )\{
\}
 match_kind\{
 exact , ternary , lpm\}
 match_kind\{
 range , optional , selector\}
 const bit \< 32 \> __v1model_version = 20180101;
 @ metadata @ name ( \"standard_metadata\" ) struct standard_metadata_t\{
 bit \< 9 \> ingress_port;
 bit \< 9 \> egress_spec;
 bit \< 9 \> egress_port;
 bit \< 32 \> instance_type;
 bit \< 32 \> packet_length;
 @ alias ( \"queueing_metadata.enq_timestamp\" ) bit \< 32 \> enq_timestamp;
 @ alias ( \"queueing_metadata.enq_qdepth\" ) bit \< 19 \> enq_qdepth;
 @ alias ( \"queueing_metadata.deq_timedelta\" ) bit \< 32 \> deq_timedelta;
 @ alias ( \"queueing_metadata.deq_qdepth\" ) bit \< 19 \> deq_qdepth;
 @ alias ( \"intrinsic_metadata.ingress_global_timestamp\" ) bit \< 48 \> ingress_global_timestamp;
 @ alias ( \"intrinsic_metadata.egress_global_timestamp\" ) bit \< 48 \> egress_global_timestamp;
 @ alias ( \"intrinsic_metadata.mcast_grp\" ) bit \< 16 \> mcast_grp;
 @ alias ( \"intrinsic_metadata.egress_rid\" ) bit \< 16 \> egress_rid;
 bit \< 1 \> checksum_error;
 error parser_error;
 @ alias ( \"intrinsic_metadata.priority\" ) bit \< 3 \> priority;
\}
 enum CounterType\{
 packets , bytes , packets_and_bytes\}
 enum MeterType\{
 packets , bytes\}
 extern counter\{
 counter ( bit \< 32 \> size , CounterType type );
 void count ( in bit \< 32 \> index );
\}
 extern direct_counter\{
 direct_counter ( CounterType type );
 void count ( );
\}
 extern meter\{
 meter ( bit \< 32 \> size , MeterType type );
 void execute_meter \< T \> ( in bit \< 32 \> index , out T result );
\}
 extern direct_meter \< T \>\{
 direct_meter ( MeterType type );
 void read ( out T result );
\}
 extern register \< T \>\{
 register ( bit \< 32 \> size );
 @ noSideEffects void read ( out T result , in bit \< 32 \> index );
 void write ( in bit \< 32 \> index , in T value );
\}
 extern action_profile\{
 action_profile ( bit \< 32 \> size );
\}
 extern void random \< T \> ( out T result , in T lo , in T hi );
 extern void digest \< T \> ( in bit \< 32 \> receiver , in T data );
 enum HashAlgorithm\{
 crc32 , crc32_custom , crc16 , crc16_custom , random , identity , csum16 , xor16\}
 @ deprecated ( \"Please use mark_to_drop(standard_metadata) instead.\" ) extern void mark_to_drop ( );
 @ pure extern void mark_to_drop ( inout standard_metadata_t standard_metadata );
 @ pure extern void hash \< O , T , D , M \> ( out O result , in HashAlgorithm algo , in T base , in D data , in M max );
 extern action_selector\{
 action_selector ( HashAlgorithm algorithm , bit \< 32 \> size , bit \< 32 \> outputWidth );
\}
 enum CloneType\{
 I2E , E2E\}
 @ deprecated ( \"Please use verify_checksum/update_checksum instead.\" ) extern Checksum16\{
 Checksum16 ( );
 bit \< 16 \> get \< D \> ( in D data );
\}
 extern void verify_checksum \< T , O \> ( in bool condition , in T data , in O checksum , HashAlgorithm algo );
 @ pure extern void update_checksum \< T , O \> ( in bool condition , in T data , inout O checksum , HashAlgorithm algo );
 extern void verify_checksum_with_payload \< T , O \> ( in bool condition , in T data , in O checksum , HashAlgorithm algo );
 @ noSideEffects extern void update_checksum_with_payload \< T , O \> ( in bool condition , in T data , inout O checksum , HashAlgorithm algo );
 extern void resubmit \< T \> ( in T data );
 extern void recirculate \< T \> ( in T data );
 extern void clone ( in CloneType type , in bit \< 32 \> session );
 extern void clone3 \< T \> ( in CloneType type , in bit \< 32 \> session , in T data );
 extern void truncate ( in bit \< 32 \> length );
 extern void assert ( in bool check );
 extern void assume ( in bool check );
 extern void log_msg ( string msg );
 extern void log_msg \< T \> ( string msg , in T data );
 parser Parser \< H , M \> ( packet_in b , out H parsedHdr , inout M meta , inout standard_metadata_t standard_metadata );
 control VerifyChecksum \< H , M \> ( inout H hdr , inout M meta );
 @ pipeline control Ingress \< H , M \> ( inout H hdr , inout M meta , inout standard_metadata_t standard_metadata );
 @ pipeline control Egress \< H , M \> ( inout H hdr , inout M meta , inout standard_metadata_t standard_metadata );
 control ComputeChecksum \< H , M \> ( inout H hdr , inout M meta );
 @ deparser control Deparser \< H \> ( packet_out b , in H hdr );
 package V1Switch \< H , M \> ( Parser \< H , M \> p , VerifyChecksum \< H , M \> vr , Ingress \< H , M \> ig , Egress \< H , M \> eg , ComputeChecksum \< H , M \> ck , Deparser \< H \> dep );
 const bit \< 16 \> TYPE_IPV4 = 0x800;
 typedef bit \< 9 \> egressSpec_t;
 typedef bit \< 48 \> macAddr_t;
 typedef bit \< 32 \> ip4Addr_t;
 header ethernet_t\{
 macAddr_t destinationAddress;
 macAddr_t sourceAddress;
 macAddr_t dstAddr;
 macAddr_t srcAddr;
 bit \< 16 \> etherType;
}
 header ipv4_t\{
 ip4Addr_t destinationAddress;
 ip4Addr_t sourceAddress;
 bit \< 4 \> version;
 bit \< 4 \> ihl;
 bit \< 8 \> diffserv;
 bit \< 16 \> totalLen;
 bit \< 16 \> identification;
 bit \< 3 \> flags;
 bit \< 13 \> fragOffset;
 bit \< 8 \> ttl;
 bit \< 8 \> protocol;
 bit \< 16 \> hdrChecksum;
 ip4Addr_t srcAddr;
 ip4Addr_t dstAddr;
}
 struct metadata\{
\}
 struct headers\{
 ethernet_t ethernet;
 ipv4_t ipv4;
 ethernet_t ethernet2;
 ipv4_t ipv42;
 ethernet_t ethernet3;
 ipv4_t ipv43;
 ethernet_t ethernet4;
 ipv4_t ipv44;
\}
 parser MyParser ( packet_in packet , out headers hdr , inout metadata meta , inout standard_metadata_t standard_metadata )\{
 state start\{
 transition parse_ethernet;
\}
 state parse_ethernet\{
 packet.extract ( hdr.ethernet );
 transition select ( hdr.ethernet.etherType )\{
 TYPE_IPV4 : parse_ipv4;
 default : accept;
\}
\}
 state parse_ipv4\{
 packet.extract ( hdr.ipv4 );
 transition parse_ethernet2;
\}
 state parse_ethernet2\{
 packet.extract ( hdr.ethernet2 );
 transition select ( hdr.ethernet2.etherType )\{
 TYPE_IPV4 : parse_ipv42;
 default : accept;
\}
\}
 state parse_ipv42\{
 packet.extract ( hdr.ipv42 );
 transition parse_ethernet3;
\}
 state parse_ethernet3\{
 packet.extract ( hdr.ethernet3 );
 transition select ( hdr.ethernet3.etherType )\{
 TYPE_IPV4 : parse_ipv43;
 default : accept;
\}
\}
 state parse_ipv43\{
 packet.extract ( hdr.ipv43 );
 transition parse_ethernet4;
\}
 state parse_ethernet4\{
 packet.extract ( hdr.ethernet4 );
 transition select ( hdr.ethernet4.etherType )\{
 TYPE_IPV4 : parse_ipv44;
 default : accept;
\}
\}
 state parse_ipv44\{
 packet.extract ( hdr.ipv44 );
 transition accept;
\}
\}
 control MyVerifyChecksum ( inout headers hdr , inout metadata meta )\{
 apply\{
\}
\}
 control MyIngress ( inout headers hdr , inout metadata meta , inout standard_metadata_t standard_metadata )\{
 action drop ( )\{
 mark_to_drop ( standard_metadata );
\}
 action ipv4_forward ( macAddr_t dstAddr , egressSpec_t port )\{
 standard_metadata.egress_spec = port;
 hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;
 hdr.ethernet.dstAddr = dstAddr;
 hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
\}
 action ipv42_forward ( macAddr_t dstAddr , egressSpec_t port )\{
 standard_metadata.egress_spec = port;
 hdr.ethernet2.srcAddr = hdr.ethernet2.dstAddr;
 hdr.ethernet2.dstAddr = dstAddr;
 hdr.ipv42.ttl = hdr.ipv42.ttl - 1;
\}
 action ipv43_forward ( macAddr_t dstAddr , egressSpec_t port )\{
 standard_metadata.egress_spec = port;
 hdr.ethernet3.srcAddr = hdr.ethernet3.dstAddr;
 hdr.ethernet3.dstAddr = dstAddr;
 hdr.ipv43.ttl = hdr.ipv43.ttl - 1;
\}
 action ipv44_forward ( macAddr_t dstAddr , egressSpec_t port )\{
 standard_metadata.egress_spec = port;
 hdr.ethernet4.srcAddr = hdr.ethernet4.dstAddr;
 hdr.ethernet4.dstAddr = dstAddr;
 hdr.ipv44.ttl = hdr.ipv44.ttl - 1;
\}
 table ipv4_lpm\{
 key =\{
 hdr.ipv4.dstAddr : lpm;
\}
 actions =\{
 ipv4_forward;
 drop;
 NoAction;
\}
 size = 1024;
 default_action = drop ( );
\}
 table ipv42_lpm\{
 key =\{
 hdr.ipv42.dstAddr : lpm;
\}
 actions =\{
 ipv42_forward;
 drop;
 NoAction;
\}
 size = 1024;
 default_action = drop ( );
\}
 table ipv43_lpm\{
 key =\{
 hdr.ipv43.dstAddr : lpm;
\}
 actions =\{
 ipv43_forward;
 drop;
 NoAction;
\}
 size = 1024;
 default_action = drop ( );
\}
 table ipv44_lpm\{
 key =\{
 hdr.ipv44.dstAddr : lpm;
\}
 actions =\{
 ipv44_forward;
 drop;
 NoAction;
\}
 size = 1024;
 default_action = drop ( );
\}
 apply\{
 if ( hdr.ipv4.isValid ( ) )\{
 ipv4_lpm.apply ( );
 if ( hdr.ipv42.isValid ( ) )\{
 ipv42_lpm.apply ( );
 if ( hdr.ipv43.isValid ( ) )\{
 ipv43_lpm.apply ( );
 if ( hdr.ipv44.isValid ( ) )\{
 ipv44_lpm.apply ( );
\}
 else\{
\}
\}
 else\{
\}
\}
 else\{
\}
\}
 else\{
\}
\}
\}
 control MyEgress ( inout headers hdr , inout metadata meta , inout standard_metadata_t standard_metadata )\{
 apply\{
\}
\}
 control MyComputeChecksum ( inout headers hdr , inout metadata meta )\{
 apply\{
 update_checksum ( hdr.ipv4.isValid ( ) ,\{
 hdr.ipv4.version , hdr.ipv4.ihl , hdr.ipv4.diffserv , hdr.ipv4.totalLen , hdr.ipv4.identification , hdr.ipv4.flags , hdr.ipv4.fragOffset , hdr.ipv4.ttl , hdr.ipv4.protocol , hdr.ipv4.srcAddr , hdr.ipv4.dstAddr\}
 , hdr.ipv4.hdrChecksum , HashAlgorithm.csum16 );
\}
\}
 control MyDeparser ( packet_out packet , in headers hdr )\{
 apply\{
 packet.emit ( hdr.ethernet );
 packet.emit ( hdr.ipv4 );
 packet.emit ( hdr.ethernet2 );
 packet.emit ( hdr.ipv42 );
 packet.emit ( hdr.ethernet3 );
 packet.emit ( hdr.ipv43 );
 packet.emit ( hdr.ethernet4 );
 packet.emit ( hdr.ipv44 );
\}
\}
 V1Switch ( MyParser ( ) , MyVerifyChecksum ( ) , MyIngress ( ) , MyEgress ( ) , MyComputeChecksum ( ) , MyDeparser ( ) ) main;

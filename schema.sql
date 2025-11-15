--
-- PostgreSQL database dump
--

-- Dumped from database version 17.5
-- Dumped by pg_dump version 17.5

-- Started on 2025-11-15 16:11:06

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 218 (class 1259 OID 40962)
-- Name: hospital_feedback; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hospital_feedback (
    id integer NOT NULL,
    feedback_text text NOT NULL,
    sentiment_pricing integer,
    sentiment_appointments integer,
    sentiment_staff integer,
    sentiment_customer_service integer,
    sentiment_emergency_services integer,
    doctor_name text,
    staff_role text,
    hospital_name text,
    department text,
    specialty text,
    service_area text,
    price text,
    time_expression text,
    location text,
    quality_aspect text,
    issue_type text,
    treatment_type text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.hospital_feedback OWNER TO postgres;

--
-- TOC entry 217 (class 1259 OID 40961)
-- Name: hospital_feedback_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.hospital_feedback_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.hospital_feedback_id_seq OWNER TO postgres;

--
-- TOC entry 4795 (class 0 OID 0)
-- Dependencies: 217
-- Name: hospital_feedback_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.hospital_feedback_id_seq OWNED BY public.hospital_feedback.id;


--
-- TOC entry 4641 (class 2604 OID 40965)
-- Name: hospital_feedback id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hospital_feedback ALTER COLUMN id SET DEFAULT nextval('public.hospital_feedback_id_seq'::regclass);


--
-- TOC entry 4644 (class 2606 OID 40970)
-- Name: hospital_feedback hospital_feedback_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hospital_feedback
    ADD CONSTRAINT hospital_feedback_pkey PRIMARY KEY (id);


-- Completed on 2025-11-15 16:11:06

--
-- PostgreSQL database dump complete
--

